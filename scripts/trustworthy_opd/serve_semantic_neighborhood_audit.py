from __future__ import annotations

import argparse
import json
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from common import read_jsonl
from four_group_common import atomic_write_jsonl
from semantic_neighborhood_audit_common import VALID_LABELS, latest_annotations

ANNOTATOR_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}")

PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>语义邻域盲审</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f3f4f6;color:#111827}
main{max-width:1450px;margin:auto;padding:22px}.card{background:#fff;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 1px 5px #0002}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;overflow:auto;max-height:55vh;background:#f8fafc;padding:14px;border-radius:8px;border:1px solid #dbe3ee;line-height:1.45}
button{padding:11px 17px;margin:5px;border:0;border-radius:8px;font-weight:650;cursor:pointer}.yes{background:#16a34a;color:#fff}.no{background:#dc2626;color:#fff}.maybe{background:#d97706;color:#fff}.nav{background:#475569;color:#fff}
textarea{width:100%;min-height:58px}#status{font-weight:650}.action{font-family:monospace;background:#eef2ff;padding:4px 8px;border-radius:5px}.muted{color:#64748b}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>语义邻域盲审</h1><p class="muted">只判断两个状态是否处于同一个即时推理子目标，以及 anchor 的采样动作在 neighbor 状态下是否有可比较的语义。不要判断答案最终是否正确。</p>
<section class="card"><span id="status">加载中……</span>　<button class="nav" onclick="previous()">← 上一个</button><button class="nav" onclick="nextUnlabeled()">下一个未标注 →</button></section>
<section class="card"><h3>题目</h3><pre id="prompt"></pre><p>Anchor 采样动作：<span class="action" id="action"></span></p></section>
<section class="grid"><div class="card"><h2>状态 A（anchor）</h2><pre id="anchor"></pre></div><div class="card"><h2>状态 B（neighbor）</h2><pre id="neighbor"></pre></div></section>
<section class="card"><label>置信程度：<select id="confidence"><option value="3">高</option><option value="2" selected>中</option><option value="1">低</option></select></label><p><textarea id="notes" placeholder="可选备注；不要在这里推测隐藏的正确性组或指标值"></textarea></p>
<button class="yes" onclick="annotate('comparable')">1　语义可比较</button><button class="no" onclick="annotate('not_comparable')">2　不可比较</button><button class="maybe" onclick="annotate('uncertain')">3　不确定</button></section>
</main><script>
const params=new URLSearchParams(location.search);let annotator=params.get('annotator');let subset=params.get('subset')||'all';
if(!annotator){annotator=prompt('请输入标注者 ID（英文字母、数字、_、-）','liufengkai');if(annotator){params.set('annotator',annotator);history.replaceState(null,'','?'+params.toString())}}
let items=[],labels={},index=0;
function escapedAction(value){return JSON.stringify(value)}
async function load(){const response=await fetch(`/api/items?annotator=${encodeURIComponent(annotator)}&subset=${encodeURIComponent(subset)}`);if(!response.ok){alert(await response.text());return}const payload=await response.json();items=payload.items;labels=payload.labels;index=Math.max(0,items.findIndex(x=>!labels[x.audit_id]));if(index<0)index=0;render()}
function render(){if(!items.length){document.getElementById('status').textContent='该子集没有样本';return}const item=items[index];document.getElementById('prompt').textContent=item.prompt_text;document.getElementById('anchor').textContent=item.anchor_excerpt;document.getElementById('neighbor').textContent=item.neighbor_excerpt;document.getElementById('action').textContent=escapedAction(item.anchor_action);const old=labels[item.audit_id]||{};document.getElementById('confidence').value=old.confidence||2;document.getElementById('notes').value=old.notes||'';const done=items.filter(x=>labels[x.audit_id]).length;document.getElementById('status').textContent=`标注者 ${annotator}｜${subset} 子集｜${index+1}/${items.length}｜已完成 ${done}/${items.length}`}
async function annotate(label){const item=items[index];const body={annotator,audit_id:item.audit_id,label,confidence:Number(document.getElementById('confidence').value),notes:document.getElementById('notes').value};const response=await fetch('/api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!response.ok){alert(await response.text());return}labels[item.audit_id]=body;nextUnlabeled()}
function nextUnlabeled(){for(let step=1;step<=items.length;step++){const candidate=(index+step)%items.length;if(!labels[items[candidate].audit_id]){index=candidate;render();return}}index=Math.min(index+1,items.length-1);render()}
function previous(){index=(index-1+items.length)%items.length;render()}
document.addEventListener('keydown',event=>{if(event.target.tagName==='TEXTAREA'||event.target.tagName==='SELECT')return;if(event.key==='1')annotate('comparable');if(event.key==='2')annotate('not_comparable');if(event.key==='3')annotate('uncertain');if(event.key==='ArrowLeft')previous();if(event.key==='ArrowRight')nextUnlabeled()});load();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve blinded semantic audit UI")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8094, type=int)
    return parser.parse_args()


def safe_annotator(value: str) -> str:
    if not ANNOTATOR_PATTERN.fullmatch(value):
        raise ValueError("annotator must match [A-Za-z0-9_-]{1,40}")
    return value


def build_handler(run_dir: Path):
    blinded = read_jsonl(run_dir / "artifacts" / "blinded_pairs.jsonl")
    allowed_ids = {str(row["audit_id"]) for row in blinded}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_payload(
            self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_payload(
                json.dumps(value, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_payload(PAGE.encode(), "text/html; charset=utf-8")
                return
            if parsed.path != "/api/items":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                query = parse_qs(parsed.query)
                annotator = safe_annotator(query.get("annotator", [""])[0])
                subset = query.get("subset", ["all"])[0]
                if subset not in {"all", "secondary"}:
                    raise ValueError("subset must be all or secondary")
                items = (
                    blinded
                    if subset == "all"
                    else [row for row in blinded if row["secondary_required"]]
                )
                labels = latest_annotations(
                    run_dir / "annotations" / f"{annotator}.jsonl"
                )
                self.send_json({"items": items, "labels": labels})
            except ValueError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/annotate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 100_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                annotator = safe_annotator(str(payload.get("annotator", "")))
                audit_id = str(payload.get("audit_id", ""))
                label = str(payload.get("label", ""))
                confidence = int(payload.get("confidence", 2))
                notes = str(payload.get("notes", ""))
                if audit_id not in allowed_ids:
                    raise ValueError("unknown audit_id")
                if label not in VALID_LABELS:
                    raise ValueError("invalid label")
                if confidence not in {1, 2, 3}:
                    raise ValueError("confidence must be 1, 2, or 3")
                if len(notes) > 2000:
                    raise ValueError("notes is too long")
                path = run_dir / "annotations" / f"{annotator}.jsonl"
                row = {
                    "audit_id": audit_id,
                    "label": label,
                    "confidence": confidence,
                    "notes": notes,
                    "annotator": annotator,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                with lock:
                    latest = latest_annotations(path)
                    latest[audit_id] = row
                    order = {
                        str(item["audit_id"]): int(item["audit_index"])
                        for item in blinded
                    }
                    atomic_write_jsonl(
                        path,
                        sorted(
                            latest.values(), key=lambda item: order[item["audit_id"]]
                        ),
                    )
                self.send_json({"saved": True})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return Handler


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "artifacts" / "blinded_pairs.jsonl").is_file():
        raise FileNotFoundError(run_dir / "artifacts" / "blinded_pairs.jsonl")
    server = ThreadingHTTPServer((args.host, args.port), build_handler(run_dir))
    print(f"AUDIT_URL=http://{args.host}:{args.port}/?annotator=liufengkai", flush=True)
    print(f"RUN_DIR={run_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
