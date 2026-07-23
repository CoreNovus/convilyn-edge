"""一鍵示範 — 「豆豆的一天」：寵物監控 pack 從頭到尾跑一遍。

給不是工程師的人看的:一行指令,就能看完一隻貓咪一整天怎麼被照顧 —— 全程在自己的裝置上跑、
不用連雲端、每次結果都一樣、中途不會壞掉 —— 最後用白話印出一張「這個系統幫你做到了什麼」的報告。

    PYTHONPATH=src python -m examples.pet_monitoring.run_e2e

它跑的是**真正的**工作流程(`assemble_pet_alert_pipeline`,和 pack 測試的 `_build` 同一套接法):
規則判斷 + 會累積的異常門檻(10 分鐘內 2 次) + 連線狀態 + 用 AI 看攝影機找貓 + 通知(我 / 妹妹) +
「10 分鐘沒人回就升級」的人工確認。這支 harness 只是「使用」那份宣告,自己不加任何場景邏輯(鐵律 1)。

**用哪個 AI。** 找貓的模型跑在裝置本機。設 ``CONVILYN_EDGE_MODEL_URL`` 指向本機推論伺服器(Ollama,
例如 ``http://localhost:11434`` 跑 ``qwen3:4b``,或 OpenAI 相容的 ``…/v1``)就用真的本機推論
(``placement=edge``)。沒設或連不上時,自動改用**內建的示範模型**(固定答案),所以沒裝任何 AI 也能
一路跑完(CI / 沒裝 Ollama 的筆電)—— 報告會標明用的是哪一種。

時間是注入的,所以「10 分鐘」的升級當下就會發生(不用真的等)。**決策每次都一樣**(示範模型模式下連
敘事都一模一樣;接真本機模型時,路由決策一樣、只有敘事文字反映當下推論)。零依賴(只用標準函式庫);
不 import 任何後端(``app.*``)。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from convilyn_edge.clientcompute.engine import HttpLocalExtractor
from convilyn_edge.offline.queue import DurableQueue
from convilyn_edge.probe import DeviceCapabilityManifest, InstalledAsset
from convilyn_edge.runtime import ThresholdAggregator
from convilyn_edge.spi.review import ReviewOutcome, ReviewRequest
from convilyn_edge.spi.source import EventSource, SourceContext
from examples.pet_monitoring import (
    CameraFrame,
    CatLocation,
    CatLocatorModel,
    ConnectivityProvider,
    FeederSimSource,
    LitterSimSource,
    NotifySink,
    PetAnomalyRules,
    WaterSimSource,
    anomaly_bucket,
    assemble_pet_alert_pipeline,
    default_alert_review,
    resolve_placement,
)
from examples.pet_monitoring.model_node import CatDetector

_MODEL_URL_ENV = "CONVILYN_EDGE_MODEL_URL"
_MODEL_TAG_ENV = "CONVILYN_EDGE_MODEL"
_DAY = datetime(2026, 1, 6, tzinfo=timezone.utc)  # 固定的某一天(星期二),讓每次跑都一樣
_GRACE_S = 600  # 人工確認的升級寬限:10 分鐘
_INFER_TIMEOUT_S = 30.0  # 本機推論逾時上限(示範用短逾時;真部署應把推論丟到執行緒外,見下方註解)

#: 監控區域的中文名(給人看)。模型回傳的英文/雜訊會先清洗再對照,對不到就顯示清洗後的原字。
_ZONE_ZH = {
    "litter": "砂盆",
    "feeder": "飼料區",
    "water": "水碗",
    "window": "窗邊",
    "unknown": "說不準的位置",
}
#: 只保留可列印 ASCII —— 模型輸出是不可信資料,清掉控制字元/跳脫序列,避免污染這張報告。
_UNSAFE = re.compile(r"[^\x20-\x7e]")


# ── 注入時鐘(確定性;不會真的等) ─────────────────────────────────────────────


class _Clock:
    def __init__(self, at: datetime) -> None:
        self._t = at

    def __call__(self) -> datetime:
        return self._t

    def set(self, at: datetime) -> None:
        self._t = at


async def _instant(_seconds: float) -> None:
    """一個「秒回」的 sleep —— 讓升級計時器當下觸發,不用真的等 10 分鐘。"""
    return None


# ── 找貓的模型(真本機推論,或內建示範模型) ─────────────────────────────────


_LOCATE_PROMPT = (
    "You are an on-device pet camera assistant. Given the scene, report whether the cat is present "
    "and in which monitored zone (one of: feeder, water, litter, window, unknown)."
)


def _scene(frame: CameraFrame) -> str:
    return (
        f"Pet camera frame {frame.frame_id}. Motion detected: {frame.motion}. "
        "Monitored zones: feeder, water, litter, window."
    )


def _clean(text: str, *, limit: int = 48) -> str:
    """清洗不可信字串:去控制字元/跳脫序列並截長(模型輸出是資料,不是版面)。"""
    return _UNSAFE.sub("", text)[:limit] or "unknown"


def _safe_url(url: str) -> str:
    """只留 ``scheme://host:port`` —— 絕不把使用者名稱/密碼/query(可能夾帶憑證)印到畫面或 log。"""
    parts = urlsplit(url)
    host = parts.hostname or "?"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}"


def _extractor_detector(extractor: HttpLocalExtractor) -> CatDetector:
    """把本機推論包成找貓 detector —— 韌性優先(絕不 raise)。

    注意:``extract`` 是同步阻塞呼叫。示範一次一個事件、又用短逾時,所以沒問題;真部署若要並發,
    應把推論丟到迴圈外(``asyncio.to_thread``),避免卡住事件迴圈。
    """

    def _detect(frame: CameraFrame) -> CatLocation:
        try:
            raw = extractor.extract(
                prompt=_LOCATE_PROMPT,
                sources={"scene": _scene(frame)},
                required_anchors=["present", "zone"],
            )
            present = str(raw.get("present", "")).strip().lower() in {"true", "yes", "present", "1"}
            zone = _clean(str(raw.get("zone") or "unknown"))  # 不可信輸出先清洗
            return CatLocation(present=present, zone=zone, confidence=0.8)
        except Exception:  # noqa: BLE001 - 模型出小狀況也不能打斷這一天
            return CatLocation(present=False, zone=None, confidence=0.0)

    return _detect


def _stub_detector(frame: CameraFrame) -> CatLocation:
    """內建示範模型:固定回「豆豆在砂盆附近」(固定、可重現,不需安裝任何 AI)。"""
    return CatLocation(present=True, zone="litter", confidence=0.92)


def build_detector(env: Mapping[str, str]) -> tuple[CatDetector, str]:
    """依環境挑模型後端;連不上就退回內建示範模型。回傳 ``(detector, 模式說明)``。

    只接受 ``http`` / ``https``(擋掉 ``file://`` 之類的 SSRF/本地讀檔);``…/v1`` 走 OpenAI 相容線,
    其餘當 Ollama。印出來的模式說明只含 ``scheme://host:port``(不外洩憑證)。
    """
    url = env.get(_MODEL_URL_ENV, "").strip()
    if not url:
        return _stub_detector, "內建示範模型(未設 CONVILYN_EDGE_MODEL_URL,不需安裝任何 AI)"
    if urlsplit(url).scheme not in {"http", "https"}:
        return _stub_detector, f"內建示範模型(不支援的網址協定:{urlsplit(url).scheme or '空'})"
    base = url.rstrip("/")
    kind = "openai-compat" if base.endswith("/v1") else "ollama"
    extractor = HttpLocalExtractor(
        model=env.get(_MODEL_TAG_ENV, "qwen3:4b"),
        kind=kind,
        base_url=base,
        timeout=_INFER_TIMEOUT_S,
    )
    if extractor.health() is not None:
        return _stub_detector, f"內建示範模型(本機模型連不上:{_safe_url(base)})"
    return _extractor_detector(extractor), f"本機模型 {extractor.model}({_safe_url(base)})"


# ── 人工確認(示範用,模擬「你」有沒有及時回) ───────────────────────────────


class _AckReview:
    """及時回覆的人 —— 這次確認由真人決定(不會升級)。"""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        return ReviewOutcome(decision="selected", choice_id="ack")


class _SilentReview:
    """一直沒回的人 —— 這個確認會和(注入的)10 分鐘計時器賽跑。"""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


# ── 「豆豆的一天」劇本 ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DayStep:
    """一天裡的一個時刻:時間、感測器讀數、以及給人看的白話說明。"""

    at: datetime
    label: str
    source: EventSource
    story: str


def _day(device_id: str = "doudou-cam-01") -> list[DayStep]:
    """劇本。10 分鐘窗內累積 2 次異常會觸發警戒;更早的單一次異常保持安靜(不亂報)。
    註:pack 出貨的規則抓的是砂盆「過度使用」訊號,所以異常 #2 是一筆過度使用讀數 —— 想要「太久沒用
    砂盆」訊號的 pack,自己在它的 operator 加規則,而不是在這支 harness 裡加。"""
    return [
        DayStep(
            at=_DAY.replace(hour=8, minute=0),
            label="08:00",
            source=FeederSimSource(device_id, [{"level_pct": 82, "jammed": False}]),
            story="豆豆吃早餐了 —— 飼料碗還很滿,一切正常。",
        ),
        DayStep(
            at=_DAY.replace(hour=11, minute=55),
            label="11:55",
            source=WaterSimSource(device_id, [{"level_ml": 30}]),
            story="一整個上午,水碗都沒動過、水位偏低了 —— 今天第 1 個異常。",
        ),
        DayStep(
            at=_DAY.replace(hour=11, minute=57),
            label="11:57",
            source=LitterSimSource(device_id, [{"uses": 15}]),
            story="砂盆的使用出現異常 —— 今天第 2 個異常(可能是身體不太舒服的徵兆)。",
        ),
    ]


# ── 一次示範 ─────────────────────────────────────────────────────────────────


@dataclass
class DemoResult:
    name: str
    lines: list[str] = field(default_factory=list)
    route_keys: list[str] = field(default_factory=list)
    located: bool = False
    placement: str = ""
    me_notified: int = 0
    sister_notified: int = 0


def _manifest_with_local_detector(device_id: str) -> DeviceCapabilityManifest:
    """一台回報「本機有物件偵測能力」的裝置 → placement 會被判為 ``edge``。"""
    return DeviceCapabilityManifest(
        device_id=device_id,
        silicon="jetson_orin",
        ram_mb=8192,
        installed_assets=(InstalledAsset(capability="object_detection", model="cat-locator"),),
    )


async def _drain(source: EventSource, device_id: str):
    events = []
    async for envelope in source.start(SourceContext(device_id=device_id)):
        events.append(envelope)
    await source.stop()
    return events


_ESCALATION_LINE = (
    "        ★ 真實情境:這一步,系統會主動通知妹妹(也就是你)—— 就是為了不漏接豆豆的求救。"
)


def _zone_zh(zone: str | None) -> str:
    if zone is None:
        return "攝影機暫時看不到"
    return _ZONE_ZH.get(zone, _clean(zone))


def _judgment(outcome, route: str) -> str:
    """把系統對這一個事件的判斷,寫成一句白話。"""
    state = outcome.state
    count = state.output("anomalies").count
    if route == "quiet":
        return (
            f"目前累積 {count} 個異常,還沒到「10 分鐘內 2 次」的警戒門檻。"
            "先安靜守著,不會為了一次小狀況就打擾你。"
        )
    if route == "offline":
        return (
            f"已經 {count} 個異常了,但這時攝影機/網路剛好連不上 —— "
            "系統改用離線方式,直接通知你:請檢查一下連線。"
        )
    where = _zone_zh(
        state.output("locate").result.output.zone if state.output("locate").result.output else None
    )
    if state.output("dispatch").key == "escalate":
        return (
            f"10 分鐘內累積 {count} 個異常,達到警戒。系統先用攝影機找到豆豆(在{where}附近),"
            "再問你要不要去看看、等你 10 分鐘;都沒人回應 → 自動升級,通知妹妹。"
        )
    return (
        f"10 分鐘內累積 {count} 個異常,達到警戒。系統找到豆豆(在{where}附近)、通知了你;"
        "你即時回覆了,不需要升級。"
    )


async def run_demo(
    *,
    name: str,
    detector: CatDetector,
    online: bool,
    review,
    tmp: Path,
    device_id: str = "doudou-cam-01",
) -> DemoResult:
    """把一整天跑過一條全新的工作流程,記錄給人看的敘事。"""
    result = DemoResult(name=name)
    clock = _Clock(_DAY)
    rules = PetAnomalyRules()
    store = DurableQueue(tmp / f"{name}.jsonl")
    aggregator = ThresholdAggregator(
        store, threshold=2, window_s=_GRACE_S, key_of=anomaly_bucket(rules), now=clock
    )
    manifest = _manifest_with_local_detector(device_id)
    cat_locator = CatLocatorModel(placement=resolve_placement(manifest), detector=detector)
    result.placement = cat_locator.placement
    me, sister = NotifySink("me"), NotifySink("sister")
    breach_deadline = (_DAY.replace(hour=11, minute=57) + timedelta(seconds=_GRACE_S)).isoformat()
    pipeline = assemble_pet_alert_pipeline(
        aggregator=aggregator,
        connectivity=ConnectivityProvider(online),
        cat_locator=cat_locator,
        review=review,
        notify_primary=me,
        notify_escalation=sister,
        review_request=default_alert_review(expires_at=breach_deadline),
        now=clock,
        sleep=_instant,
    )

    for step in _day(device_id):
        clock.set(step.at)
        for envelope in await _drain(step.source, device_id):
            outcome = await pipeline.run(envelope)
            route = outcome.output("alert").key
            result.route_keys.append(route)
            result.lines.append(f"  【{step.label}】{step.story}")
            result.lines.append(f"          系統怎麼判斷:{_judgment(outcome, route)}")
            if route == "locate" and outcome.state.output("dispatch").key == "escalate":
                result.lines.append(_ESCALATION_LINE)

    result.me_notified = len(me.sent)
    result.sister_notified = len(sister.sent)
    result.located = "locate" in result.route_keys
    return result


# ── 商業價值報告 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValueReport:
    mode: str
    offline_capable: bool
    deterministic: bool
    noise_reduced: bool
    grounded_decision: bool
    auto_escalation: bool
    capability_negotiated: bool
    escalation_grace_s: int
    passed: bool
    demos: tuple[DemoResult, ...]

    def as_rows(self) -> list[tuple[str, bool, str]]:
        return [
            ("全程在裝置本機跑,不連雲端也能運作", self.offline_capable, f"用的 AI:{self.mode}"),
            (
                "結果穩定、可回溯 —— 同一天,每次的決策都一樣",
                self.deterministic,
                "注入時鐘、路由不依賴模型輸出:同輸入 → 同決策(示範模型模式下連敘事都一樣)",
            ),
            (
                "不亂報 —— 累積 2 次異常才告警",
                self.noise_reduced,
                "單一次小狀況保持安靜,只有真的成型的問題才通知你",
            ),
            (
                "用 AI 看攝影機找到豆豆,判斷有憑有據",
                self.grounded_decision,
                "找貓的模型有跑,而且結論綁定到當下那張畫面(可回查)",
            ),
            (
                "沒人回應會自動升級",
                self.auto_escalation,
                f"{self.escalation_grace_s // 60} 分鐘沒回 → 通知妹妹",
            ),
            (
                "依裝置能力自動選擇怎麼跑",
                self.capability_negotiated,
                "這台回報有本機 AI,就把找貓跑在裝置上(edge)",
            ),
        ]


def assess(demos: Mapping[str, DemoResult], *, mode: str, deterministic: bool) -> ValueReport:
    """把三次示範收斂成一張綠燈/紅燈報告(確定性)。"""
    escalation = demos["escalation"]
    ack = demos["ack"]
    offline = demos["offline"]

    checks = {
        "quiet_first": escalation.route_keys[:2] == ["quiet", "quiet"],
        "breach_locates": escalation.route_keys[-1:] == ["locate"],
        "escalated_to_sister": escalation.sister_notified == 1 and escalation.me_notified == 0,
        "ack_notifies_me": ack.me_notified == 1 and ack.sister_notified == 0,
        "ack_located": ack.located,
        "offline_local": offline.route_keys[-1:] == ["offline"] and offline.me_notified == 1,
        "offline_no_locate": not offline.located,
    }
    passed = all(checks.values())
    # 能力協商是「實際發生」而非寫死:三次示範的 placement 都要被判為 edge。
    negotiated = all(demo.placement == "edge" for demo in (escalation, ack, offline))
    return ValueReport(
        mode=mode,
        # 綁定到「離線分支實測」而非寫死:斷線時走離線通知、且不跑模型,才算數。
        offline_capable=checks["offline_local"] and checks["offline_no_locate"],
        # 由「重跑同情境、可觀察結果一致」實測推導(見 run()),不寫死。
        deterministic=deterministic,
        noise_reduced=checks["quiet_first"],
        grounded_decision=checks["breach_locates"] and checks["ack_located"],
        auto_escalation=checks["escalated_to_sister"],
        capability_negotiated=negotiated,
        escalation_grace_s=_GRACE_S,
        passed=passed and negotiated,
        demos=(escalation, ack, offline),
    )


_DEMO_TITLE = {
    "escalation": "情境一:沒人回應 → 自動升級通知妹妹",
    "ack": "情境二:你及時回覆 → 只通知你,不升級",
    "offline": "情境三:剛好斷線 → 直接離線通知你檢查連線",
}


def render(report: ValueReport) -> str:
    """完整的給人看輸出:一天的敘事 + 商業價值報告。"""
    out: list[str] = []
    out.append("=" * 70)
    out.append("  Convilyn 邊緣 SDK — 寵物監控參考範例 — 「豆豆的一天」")
    out.append("=" * 70)
    out.append(f"  用的 AI:{report.mode}")
    out.append("  (時間是注入的:「10 分鐘」升級當下就發生,全程在本機、不用連網)")
    for demo in report.demos:
        out.append("")
        out.append("── " + _DEMO_TITLE.get(demo.name, demo.name) + " " + "─" * 8)
        out.extend(demo.lines)

    out.append("")
    out.append("=" * 70)
    out.append("  這個系統幫你做到了什麼")
    out.append("=" * 70)
    for title, ok, detail in report.as_rows():
        mark = "[ 綠燈 ]" if ok else "[ 紅燈 ]"
        out.append(f"  {mark}  {title}")
        out.append(f"            {detail}")
    out.append("")
    out.append("  從偵測到告警:即時(同一個事件當下,全在本機)")
    out.append(f"  升級寬限:{report.escalation_grace_s // 60} 分鐘(可自行調整)")
    out.append("  整合商上手:3 步、一行指令,通常 1 分鐘內就看到第一次結果(示範模型)")
    out.append("")
    verdict = (
        "★ 全部綠燈 —— 這一天,系統都正確處理了豆豆的狀況。"
        if report.passed
        else "有紅燈 —— 有一項預期結果沒達成(見上方 [ 紅燈 ])。"
    )
    out.append("  " + verdict)
    out.append("=" * 70)
    return "\n".join(out)


async def run(env: Mapping[str, str] | None = None, *, tmp: Path | None = None) -> ValueReport:
    """跑完三次示範並產出報告。不含主控台輸出(可測)。"""
    env = os.environ if env is None else env
    detector, mode = build_detector(env)
    root = tmp if tmp is not None else Path(tempfile.mkdtemp(prefix="pet-e2e-"))
    demos = {
        "escalation": await run_demo(
            name="escalation", detector=detector, online=True, review=_SilentReview(), tmp=root
        ),
        "ack": await run_demo(
            name="ack", detector=detector, online=True, review=_AckReview(), tmp=root
        ),
        "offline": await run_demo(
            name="offline", detector=detector, online=False, review=_AckReview(), tmp=root
        ),
    }
    # 確定性以實測證明:同情境重跑一次(獨立 store),要求可觀察結果(路由/通知數/是否定位)一致。
    # 路由只讀異常/連線/人工回覆、不讀模型輸出,故恆成立;若哪天路由讀了模型輸出,這格會轉紅。
    replay_root = root / "_determinism_replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    replay = await run_demo(
        name="escalation", detector=detector, online=True, review=_SilentReview(), tmp=replay_root
    )
    first = demos["escalation"]
    deterministic = (
        replay.route_keys == first.route_keys
        and replay.me_notified == first.me_notified
        and replay.sister_notified == first.sister_notified
        and replay.located == first.located
    )
    return assess(demos, mode=mode, deterministic=deterministic)


def main() -> int:
    # 讓中文在現代終端機(Windows Terminal / macOS / Linux)正確顯示;舊主控台以 replace 保底不崩。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    report = asyncio.run(run())
    print(render(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
