"""Generated decision families for the Bobcat Flash distillation corpus (2026-09-26).

Every family builds requests on the System One wire shape (`protocol.parse_request`) in
Korean and English. Some have an exact gold answer from the generator (games and rule-based
workflow checks); the rest are labelled only by the teacher (Bobcat 1 probabilities).
Rules kept here:
  * no tool-call review questions (the held-out task stays held out);
  * no TypeSafe, SemIf or Every rows, wording or cases; public text comes from the prepared
    corpora (KLUE / KorNLI / NSMC / MASSIVE / Banking77 / BoolQ rows), whose text keys the
    corpus builder checks against every evaluation split;
  * candidate order is shuffled with the row seed, so a label never sits at a fixed index.
Rows carry `supervision` = "hard_label" (gold + teacher) or "teacher_only".
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque

from bobcat.protocol import parse_request
from bobcat.schema import json_hash

MODEL = "bobcat-latest"


def seeded(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16], 16))


def row(*, family: str, task: str, lang: str, state, qid: str, question: dict,
        target: str | None, group: str, text_keys=(), score_target=None,
        basis: str = "generator_rule") -> dict:
    request = {"model": MODEL, "state": state, "questions": {qid: question}}
    _, parsed = parse_request(request)
    labels = list(parsed[0].labels)
    if target is not None and target not in labels:
        raise ValueError(f"{family}: target {target!r} not offered")
    kind = {"choice": "choice", "noul": "boolean", "score": "ordinal"}[question["type"]]
    supervision = "hard_label" if target is not None else "teacher_only"
    body = {
        "task": task, "family": family, "kind": kind, "language": lang,
        "language_origin": "generated", "group_id": group, "request": request,
        "candidate_ids": labels, "target": target, "score_target": score_target,
        "supervision": supervision, "label_basis": basis if target is not None else "teacher",
        "text_keys": sorted(set(text_keys)), "mixture_source": "generated",
    }
    body["id"] = f"flash:{family}:{json_hash([group, qid, request])[:24]}"
    return body


def shuffled(rng: random.Random, items):
    items = list(items)
    rng.shuffle(items)
    return items


# ------------------------------------------------------------------------------ games

TTT_NAMES = {
    "en": ["top-left", "top-middle", "top-right", "middle-left", "center", "middle-right",
           "bottom-left", "bottom-middle", "bottom-right"],
    "ko": ["왼쪽 위", "가운데 위", "오른쪽 위", "왼쪽 가운데", "정가운데", "오른쪽 가운데",
           "왼쪽 아래", "가운데 아래", "오른쪽 아래"],
}
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def ttt_winner(board):
    for a, b, c in LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


def ttt_rule_move(board, me):
    """Stated policy: win now; else block the opponent's immediate win; else center; else a
    corner (lowest index); else a side (lowest index)."""
    other = "O" if me == "X" else "X"
    empty = [i for i, v in enumerate(board) if v == "."]
    for player in (me, other):
        wins = []
        for i in empty:
            trial = list(board)
            trial[i] = player
            if ttt_winner(trial) == player:
                wins.append(i)
        if wins:
            return wins[0] if len(wins) == 1 else None  # ambiguous: skip the board
    if 4 in empty:
        return 4
    for group in ((0, 2, 6, 8), (1, 3, 5, 7)):
        for i in group:
            if i in empty:
                return i
    return None


def tictactoe(index: int, lang: str) -> list[dict]:
    rng = seeded("ttt", index, lang)
    for _ in range(50):
        board = ["."] * 9
        moves = rng.randint(2, 6)
        turn = "X"
        for _ in range(moves):
            empty = [i for i, v in enumerate(board) if v == "."]
            board[rng.choice(empty)] = turn
            turn = "O" if turn == "X" else "X"
        if ttt_winner(board) is None and "." in board:
            break
    else:
        return []
    me = turn
    names = TTT_NAMES[lang]
    grid = "\n".join(" ".join(board[r * 3:(r + 1) * 3]) for r in range(3))
    state = ({"game": "tic-tac-toe", "board": grid, "legend": "X, O = marks; . = empty",
              "to_move": me} if lang == "en" else
             {"게임": "틱택토", "판": grid, "범례": "X, O = 표시, . = 빈칸", "둘 차례": me})
    empty = [i for i, v in enumerate(board) if v == "."]
    group = f"flash:ttt:{index}:{lang}"
    rows = []
    move = ttt_rule_move(board, me)
    if move is not None and len(empty) >= 2:
        policy = ("Play by this policy: take a winning cell if one exists; otherwise block the "
                  "opponent's immediate win; otherwise take the center; otherwise the first free "
                  "corner in reading order; otherwise the first free side."
                  if lang == "en" else
                  "다음 정책으로 둔다: 이기는 칸이 있으면 그 칸, 없으면 상대의 즉시 승리를 막는 칸, "
                  "없으면 정가운데, 없으면 읽는 순서로 첫 빈 모서리, 없으면 첫 빈 변.")
        criteria = {names[i]: (f"place {me} at {names[i]}" if lang == "en"
                               else f"{names[i]}에 {me}를 둔다") for i in shuffled(rng, empty)}
        rows.append(row(family="game_tictactoe_move", task="flash_game", lang=lang, state=state,
                        qid="move", question={"type": "choice", "instructions": policy,
                                              "criteria": criteria},
                        target=names[move], group=group))
    wins_now = any(ttt_winner([*board[:i], me, *board[i + 1:]]) == me for i in empty)
    rows.append(row(family="game_tictactoe_win_now", task="flash_game", lang=lang, state=state,
                    qid="can_win",
                    question={"type": "noul", "instructions": (
                        f"Can {me} win on this move?" if lang == "en"
                        else f"{me}가 이번 수에 바로 이길 수 있는가?"),
                        "criteria": None},
                    target="yes" if wins_now else "no", group=group))
    return rows


DIRS = {"en": {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)},
        "ko": {"위": (-1, 0), "아래": (1, 0), "왼쪽": (0, -1), "오른쪽": (0, 1)}}


def bfs(grid, start, goal):
    size = len(grid)
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return dist


def gridworld(index: int, lang: str) -> list[dict]:
    rng = seeded("grid", index, lang)
    size = rng.randint(5, 8)
    for _ in range(50):
        grid = [["#" if rng.random() < 0.22 else "." for _ in range(size)] for _ in range(size)]
        cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == "."]
        if len(cells) < 4:
            continue
        start, goal = rng.sample(cells, 2)
        dist = bfs(grid, start, goal)
        if start not in dist or dist[start] < 2:
            continue
        good = []
        for name, (dr, dc) in DIRS[lang].items():
            nxt = (start[0] + dr, start[1] + dc)
            if nxt in dist and dist[nxt] == dist[start] - 1:
                good.append(name)
        if len(good) == 1:
            break
    else:
        return []
    grid[start[0]][start[1]], grid[goal[0]][goal[1]] = "A", "G"
    text = "\n".join("".join(r) for r in grid)
    state = ({"map": text, "legend": "A = agent, G = goal, # = wall, . = floor",
              "coordinates": "row 1 is the top row"} if lang == "en" else
             {"지도": text, "범례": "A = 에이전트, G = 목표, # = 벽, . = 바닥",
              "좌표": "1행이 맨 위"})
    criteria = {name: (f"move one cell {name}" if lang == "en" else f"{name}로 한 칸 이동")
                for name in shuffled(rng, DIRS[lang])}
    return [row(family="game_grid_shortest", task="flash_game", lang=lang, state=state,
                qid="step", question={"type": "choice", "instructions": (
                    "Which single move starts a shortest path from A to G without entering a wall?"
                    if lang == "en" else
                    "벽을 지나지 않고 A에서 G까지 가는 최단 경로의 첫 이동은 무엇인가?"),
                    "criteria": criteria}, target=good[0], group=f"flash:grid:{index}:{lang}")]


SHOOTER_STRATEGIES = {
    "survive": ("Survival. 1) If health is below 40: pick up a visible medkit, else retreat. "
                "2) Otherwise, if an enemy is closer than 10 m and ammo is above 0: attack. "
                "3) Otherwise: move forward.",
                "생존. 1) 체력이 40 미만이면 보이는 구급상자를 줍고, 없으면 후퇴한다. 2) 아니면 10m보다 "
                "가까운 적이 있고 탄약이 0보다 많을 때 공격한다. 3) 그 외에는 전진한다."),
    "aggressive": ("Aggressive. 1) If any enemy is visible and ammo is above 0: attack. 2) If ammo "
                   "is 0: pick up visible ammo, else retreat. 3) Otherwise: move forward.",
                   "공격형. 1) 적이 보이고 탄약이 0보다 많으면 공격한다. 2) 탄약이 0이면 보이는 탄약을 "
                   "줍고, 없으면 후퇴한다. 3) 그 외에는 전진한다."),
    "pacifist": ("Pacifist. Never attack. 1) If an enemy is closer than 15 m: retreat. "
                 "2) Otherwise: move forward.",
                 "평화주의. 절대 공격하지 않는다. 1) 15m보다 가까운 적이 있으면 후퇴한다. "
                 "2) 그 외에는 전진한다."),
    "collector": ("Collector. 1) If an item is visible and closer than every enemy: pick up the "
                  "nearest item. 2) Otherwise, if an enemy is visible: attack when ammo is above 5, "
                  "else retreat. 3) Otherwise: move forward.",
                  "수집형. 1) 아이템이 보이고 모든 적보다 가까우면 가장 가까운 아이템을 줍는다. 2) 아니면 "
                  "적이 보일 때 탄약이 5보다 많으면 공격하고, 아니면 후퇴한다. 3) 그 외에는 전진한다."),
}
SHOOTER_ACTIONS = {
    "en": {"attack": "fire at the nearest enemy", "retreat": "back away from enemies",
           "pick_up_medkit": "move to the visible medkit", "pick_up_ammo":
           "move to the visible ammo box", "move_forward": "advance and explore"},
    "ko": {"공격": "가장 가까운 적에게 사격", "후퇴": "적에게서 물러남",
           "구급상자 줍기": "보이는 구급상자로 이동", "탄약 줍기": "보이는 탄약 상자로 이동",
           "전진": "앞으로 나아가 탐색"},
}
ACTION_KEYS = ["attack", "retreat", "pick_up_medkit", "pick_up_ammo", "move_forward"]


def shooter_rule(strategy, health, ammo, enemies, items):
    """Exactly the numbered rules of the stated strategy."""
    nearest = min((e["distance_m"] for e in enemies), default=None)
    kinds = {i["type"]: i["distance_m"] for i in items}
    if strategy == "survive":
        if health < 40:
            return "pick_up_medkit" if "medkit" in kinds else "retreat"
        if nearest is not None and nearest < 10 and ammo > 0:
            return "attack"
        return "move_forward"
    if strategy == "aggressive":
        if nearest is not None and ammo > 0:
            return "attack"
        if ammo == 0:
            return "pick_up_ammo" if "ammo" in kinds else "retreat"
        return "move_forward"
    if strategy == "pacifist":
        return "retreat" if nearest is not None and nearest < 15 else "move_forward"
    if strategy == "collector":
        closest = min(kinds.items(), key=lambda kv: kv[1], default=None)
        if closest and (nearest is None or closest[1] < nearest):
            if len([d for d in kinds.values() if d == closest[1]]) > 1:
                return None  # tie between items: ambiguous, skip
            return "pick_up_medkit" if closest[0] == "medkit" else "pick_up_ammo"
        if nearest is not None:
            return "attack" if ammo > 5 else "retreat"
        return "move_forward"
    raise ValueError(strategy)


def shooter(index: int, lang: str) -> list[dict]:
    rng = seeded("shooter", index, lang)
    strategy = rng.choice(sorted(SHOOTER_STRATEGIES))
    health, ammo = rng.randint(5, 100), rng.choice([0, 0, 1, 3, 6, 12, 30])
    enemies = [{"type": rng.choice(["imp", "zombie", "demon", "cacodemon"]),
                "distance_m": rng.randint(3, 40), "bearing": rng.choice(["left", "ahead", "right"])}
               for _ in range(rng.choice([0, 1, 1, 2, 3]))]
    items = [{"type": t, "distance_m": rng.randint(2, 30)}
             for t in rng.sample(["medkit", "ammo"], rng.choice([0, 1, 1, 2]))]
    action = shooter_rule(strategy, health, ammo, enemies, items)
    if action is None:
        return []
    names = SHOOTER_ACTIONS[lang]
    keys = list(names)
    offered = [k for k in ACTION_KEYS
               if not (k == "pick_up_medkit" and all(i["type"] != "medkit" for i in items))
               and not (k == "pick_up_ammo" and all(i["type"] != "ammo" for i in items))]
    if action not in offered:
        return []
    labels = [keys[ACTION_KEYS.index(k)] for k in offered]
    text = SHOOTER_STRATEGIES[strategy][0 if lang == "en" else 1]
    state = ({"strategy": text, "observation": {"health": health, "ammo": ammo,
                                                "enemies": enemies, "items": items}}
             if lang == "en" else
             {"전략": text, "관측": {"체력": health, "탄약": ammo, "적": enemies, "아이템": items}})
    question = {"type": "choice", "instructions": (
        "Choose the next action that follows the strategy exactly." if lang == "en"
        else "전략을 정확히 따르는 다음 행동을 고르라."),
        "criteria": {label: names[label] for label in shuffled(rng, labels)}}
    return [row(family="game_shooter_strategy", task="flash_game", lang=lang, state=state,
                qid="action", question=question, target=keys[ACTION_KEYS.index(action)],
                group=f"flash:shooter:{index}:{lang}")]


def linkrace(index: int, lang: str, titles: list[str]) -> list[dict]:
    """Wikiracing-style link choice over real page titles; semantic, teacher-labelled."""
    if len(titles) < 20:
        return []
    rng = seeded("linkrace", index, lang)
    picked = rng.sample(titles, rng.choice([6, 8, 12, 16]) + 2)
    current, target, links = picked[0], picked[1], picked[2:]
    state = ({"current_page": current, "target_page": target, "links_on_page": links}
             if lang == "en" else {"현재 문서": current, "목표 문서": target, "문서의 링크": links})
    question = {"type": "choice", "instructions": (
        "Which link most likely brings the reader closer to the target page?" if lang == "en"
        else "어느 링크가 목표 문서에 가장 가까워질 가능성이 큰가?"),
        "criteria": {f"L{i + 1}": title for i, title in enumerate(links)}}
    return [row(family="game_linkrace", task="flash_game", lang=lang, state=state, qid="link",
                question=question, target=None, group=f"flash:linkrace:{index}:{lang}")]


# -------------------------------------------------------------------------- workflows

VENDORS = ["Acme Supply", "Northwind Traders", "Blue Harbor Logistics", "Kite Analytics",
           "Hanbit Office", "Seoul Parts", "Daehan Printing", "Orion Cloud", "Pine Labs",
           "Minjung Foods", "Jeju Water", "Atlas Hardware"]
ITEMS = ["laptop stand", "printer toner", "cloud storage (monthly)", "office chairs",
         "network switch", "consulting hours", "catering", "shipping", "software license",
         "cleaning service", "paper A4 box", "monitor 27in"]


def invoice(index: int, lang: str) -> list[dict]:
    rng = seeded("invoice", index, lang)
    approved = rng.sample(VENDORS, 6)
    vendor = rng.choice(approved if rng.random() < 0.6 else VENDORS)
    lines = []
    for _ in range(rng.randint(1, 5)):
        qty, price = rng.randint(1, 20), rng.choice([12, 45, 99, 250, 480, 1200, 3500])
        lines.append({"item": rng.choice(ITEMS), "qty": qty, "unit_price": price,
                      "amount": qty * price})
    subtotal = sum(line["amount"] for line in lines)
    tax = round(subtotal * 0.1)
    stated = subtotal + tax + (rng.choice([0, 0, 0, 10, -100, 1000]))
    po = rng.choice([None, f"PO-{rng.randint(10000, 99999)}"])
    currency = rng.choice(["USD", "USD", "KRW", "EUR"])
    invoice_doc = {"invoice_id": f"INV-{index:06d}", "vendor": vendor, "currency": currency,
                   "po_number": po, "lines": lines, "subtotal": subtotal, "tax": tax,
                   "total": stated, "due_in_days": rng.choice([7, 14, 30, 45, 60])}
    limit = rng.choice([2000, 5000, 10000])
    policy_en = (f"Approved vendors: {', '.join(approved)}. Invoices above {limit} {currency} "
                 f"need a PO number. Totals must equal subtotal plus tax.")
    policy_ko = (f"승인 거래처: {', '.join(approved)}. {limit} {currency}를 넘는 청구서는 발주번호가 "
                 f"있어야 한다. 합계는 소계와 세금의 합과 같아야 한다.")
    state = {"policy": policy_en if lang == "en" else policy_ko, "invoice": invoice_doc}
    group = f"flash:invoice:{index}:{lang}"
    consistent = stated == subtotal + tax
    on_list = vendor in approved
    needs_po = stated > limit
    route = ("reject" if not on_list else "hold_for_po" if needs_po and po is None
             else "review_total" if not consistent else "approve")
    route_names = ({"approve": "pay as scheduled", "hold_for_po": "hold until a PO number is "
                    "attached", "review_total": "send back: totals do not add up",
                    "reject": "reject: vendor not approved"} if lang == "en" else
                   {"approve": "예정대로 지급", "hold_for_po": "발주번호가 붙을 때까지 보류",
                    "review_total": "반송: 합계 불일치", "reject": "거절: 미승인 거래처"})
    rows = [
        row(family="workflow_invoice_vendor", task="flash_workflow", lang=lang, state=state,
            qid="vendor_ok", question={"type": "noul", "instructions": (
                "Is the vendor on the approved list?" if lang == "en"
                else "거래처가 승인 목록에 있는가?"), "criteria": None},
            target="yes" if on_list else "no", group=group),
        row(family="workflow_invoice_route", task="flash_workflow", lang=lang, state=state,
            qid="route", question={"type": "choice", "instructions": (
                "Apply the policy in order (vendor, PO, totals) and choose the routing."
                if lang == "en" else "정책을 순서대로(거래처, 발주번호, 합계) 적용해 처리 경로를 고르라."),
                "criteria": {k: route_names[k] for k in shuffled(rng, route_names)}},
            target=route, group=group),
        row(family="workflow_invoice_po", task="flash_workflow", lang=lang, state=state,
            qid="has_po", question={"type": "noul", "instructions": (
                "Does the invoice carry a purchase-order number?" if lang == "en"
                else "청구서에 발주번호가 있는가?"), "criteria": None},
            target="yes" if po else "no", group=group),
    ]
    return rows


BANKING_GROUPS = {
    "card": "card delivery, activation, limits, declines, lost or stolen cards",
    "transfer": "sending money, transfer status, beneficiaries, fees on transfers",
    "account": "opening, closing, verifying identity, personal details, passcode",
    "top_up": "adding money, top-up methods, top-up failures",
    "refund_dispute": "refunds, disputed or unrecognised payments, cancelled transactions",
    "exchange": "exchange rates, foreign currencies, fiat and crypto",
    "cash": "ATM withdrawals, cash, wrong amounts from an ATM",
}


def banking_group(label: str) -> str | None:
    text = label.lower()
    rules = [("refund", "refund_dispute"), ("dispute", "refund_dispute"),
             ("not_recognised", "refund_dispute"), ("reverted", "refund_dispute"),
             ("cancel", "refund_dispute"), ("atm", "cash"), ("cash", "cash"),
             ("exchange", "exchange"), ("currency", "exchange"), ("fiat", "exchange"),
             ("crypto", "exchange"), ("top_up", "top_up"), ("topping", "top_up"),
             ("transfer", "transfer"), ("beneficiary", "transfer"), ("card", "card"),
             ("pin", "card"), ("contactless", "card"), ("passcode", "account"),
             ("identity", "account"), ("verify", "account"), ("account", "account"),
             ("personal_details", "account"), ("terminate", "account")]
    for needle, group in rules:
        if needle in text:
            return group
    return None


def ticket(index: int, source: dict) -> list[dict]:
    """Customer-service triage over a real public utterance (Banking77 English, or Korean
    MASSIVE / NSMC text). Routing is gold where the source label maps to a queue; the rest
    is teacher-labelled."""
    rng = seeded("ticket", index, source["id"])
    lang = source["language"]
    tier = rng.choice(["free", "standard", "premium"])
    waited = rng.choice([1, 5, 30, 120, 600])
    prior = rng.randint(0, 4)
    state = ({"ticket": {"message": source["text"], "customer_tier": tier,
                         "minutes_waiting": waited, "previous_contacts_this_week": prior},
              "escalation_policy": "Escalate when the customer is premium and has waited more "
                                   "than 60 minutes, or has contacted us 3 or more times this week."}
             if lang == "en" else
             {"문의": {"메시지": source["text"], "고객 등급": tier, "대기 분": waited,
                     "이번 주 이전 문의 수": prior},
              "상향 정책": "프리미엄 고객이 60분 넘게 기다렸거나 이번 주 3회 이상 문의했으면 상향한다."})
    group = source["group_id"]
    keys = source.get("text_keys", [])
    escalate = (tier == "premium" and waited > 60) or prior >= 3
    rows = [row(family="workflow_ticket_escalation", task="flash_workflow", lang=lang,
                state=state, qid="escalate", question={"type": "noul", "instructions": (
                    "Does the escalation policy require escalating this ticket?" if lang == "en"
                    else "상향 정책상 이 문의를 상향해야 하는가?"), "criteria": None},
                target="yes" if escalate else "no", group=group, text_keys=keys),
            row(family="workflow_ticket_frustration", task="flash_workflow", lang=lang,
                state=state, qid="frustration", question={"type": "score", "instructions": (
                    "How frustrated does the customer sound in the message?" if lang == "en"
                    else "메시지에서 고객은 얼마나 불만스러워 보이는가?"),
                    "criteria": (["0: calm", "1: mildly annoyed", "2: frustrated",
                                  "3: angry"] if lang == "en" else
                                 ["0: 차분함", "1: 약간 불편", "2: 불만", "3: 화남"])},
                target=None, group=group, text_keys=keys),
            row(family="workflow_ticket_money_back", task="flash_workflow", lang=lang,
                state=state, qid="money_back", question={"type": "noul", "instructions": (
                    "Is the customer asking to get money back?" if lang == "en"
                    else "고객이 돈을 돌려받기를 요청하는가?"), "criteria": None},
                target=None, group=group, text_keys=keys)]
    queue = banking_group(source.get("label") or "") if source["task"] == "banking77" else None
    if queue:
        rows.append(row(family="workflow_ticket_queue", task="flash_workflow", lang=lang,
                        state=state, qid="queue", question={
                            "type": "choice", "instructions": "Route the ticket to one queue.",
                            "criteria": {k: BANKING_GROUPS[k] for k in shuffled(rng,
                                                                                BANKING_GROUPS)}},
                        target=queue, group=group, text_keys=keys, basis="banking77_label"))
    return rows


COUNTRIES = ["KR", "US", "JP", "DE", "BR", "IN", "NG", "RU", "VN", "FR"]


def security(index: int, lang: str) -> list[dict]:
    rng = seeded("security", index, lang)
    home = rng.choice(COUNTRIES[:4])
    allow = sorted(rng.sample(COUNTRIES, 3) + [home])
    events, minute = [], 0
    burst_country = rng.choice(COUNTRIES)
    for _ in range(rng.randint(2, 8)):
        minute += rng.randint(0, 9)
        events.append({"t_min": minute, "event": rng.choice(["login_failed", "login_ok",
                                                             "password_reset"]),
                       "country": rng.choice([home, home, burst_country]),
                       "mfa": rng.choice([True, False])})
    if rng.random() < 0.45:  # a burst of failures, sometimes spread too thin to count
        start, gap = rng.randint(0, 40), rng.choice([1, 1, 2, 3])
        events += [{"t_min": start + k * gap, "event": "login_failed", "country": burst_country,
                    "mfa": False} for k in range(rng.randint(4, 7))]
    if rng.random() < 0.35:
        events.append({"t_min": rng.randint(0, 60), "event": "login_ok",
                       "country": rng.choice(COUNTRIES), "mfa": rng.random() < 0.4})
    events.sort(key=lambda e: e["t_min"])
    fails = [e for e in events if e["event"] == "login_failed"]
    window = any(sum(1 for f in fails if 0 <= f["t_min"] - e["t_min"] <= 10) >= 5 for e in fails)
    foreign_ok = any(e["event"] == "login_ok" and e["country"] not in allow and not e["mfa"]
                     for e in events)
    lock = window or foreign_ok
    severity = 3 if (window and foreign_ok) else 2 if foreign_ok else 1 if window else 0
    policy = ("Lock the account if there are 5 or more failed logins within any 10 minutes, or a "
              f"successful login without MFA from a country outside the allowlist {allow}."
              if lang == "en" else
              f"10분 안에 로그인 실패가 5회 이상이거나, 허용 국가 {allow} 밖에서 MFA 없이 로그인에 "
              "성공하면 계정을 잠근다.")
    state = {"policy": policy, "account_home_country": home, "events": events}
    group = f"flash:security:{index}:{lang}"
    levels = (["0: nothing to act on", "1: brute-force attempt only",
               "2: unapproved foreign login", "3: both"] if lang == "en" else
              ["0: 조치 불필요", "1: 무차별 대입 시도만", "2: 미허용 국가 로그인", "3: 둘 다"])
    return [
        row(family="workflow_security_lock", task="flash_workflow", lang=lang, state=state,
            qid="lock", question={"type": "noul", "instructions": (
                "Does the policy require locking the account?" if lang == "en"
                else "정책상 계정을 잠가야 하는가?"), "criteria": None},
            target="yes" if lock else "no", group=group),
        row(family="workflow_security_severity", task="flash_workflow", lang=lang, state=state,
            qid="severity", question={"type": "score", "instructions": (
                "Which finding level applies?" if lang == "en" else "해당하는 탐지 수준은?"),
                "criteria": levels}, target=str(severity), group=group),
    ]


# ------------------------------------------------------------- re-questioned public rows

NLI_EN = {"supported": "The evidence establishes the claim.",
          "insufficient": "The evidence neither establishes nor contradicts the claim.",
          "contradicted": "The evidence rules the claim out."}
NLI_KO = {"지지": "근거가 주장을 뒷받침한다.", "근거 부족": "근거만으로는 주장을 판단할 수 없다.",
          "반박": "근거가 주장을 부정한다."}
NLI_MAP = {"함의": 0, "entailment": 0, "중립": 1, "neutral": 1, "모순": 2, "contradiction": 2}


def nli_pair(state) -> tuple[str, str] | None:
    """(premise, hypothesis) from the prepared NLI rows' state shapes."""
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return None
    if isinstance(state, dict) and "source_text" in state:
        return nli_pair(state["source_text"])
    if isinstance(state, dict):
        keys = list(state)
        premise = state.get("전제") or state.get("premise")
        hypothesis = state.get("가설") or state.get("hypothesis")
        if premise and hypothesis:
            return premise, hypothesis
        if len(keys) == 2 and all(isinstance(state[k], str) for k in keys):
            return state[keys[0]], state[keys[1]]
    return None


def requestion(source: dict) -> list[dict]:
    """New question wording and candidate sets over a prepared public row's own state."""
    task, lang = source["task"], source["language"]
    rng = seeded("requestion", source["id"])
    keys, group = source.get("text_keys", []), source["group_id"]
    out = []
    if task in ("kornli_multinli", "kornli_snli_1", "klue_nli"):
        pair = nli_pair(source["request"]["state"])
        target_name = source.get("target")
        if pair is None or target_name not in NLI_MAP:
            return []
        english = rng.random() < 0.5
        names = list(NLI_EN if english else NLI_KO)
        state = ({"evidence": pair[0], "claim": pair[1]} if english
                 else {"근거": pair[0], "주장": pair[1]})
        criteria = NLI_EN if english else NLI_KO
        out.append(row(family="requestion_nli_evidence", task="flash_requestion", lang=lang,
                       state=state, qid="verdict", question={"type": "choice", "instructions": (
                           "Judge the claim only from the evidence. Choose insufficient when the "
                           "evidence does not settle it." if english else
                           "근거만으로 주장을 판단하라. 근거로 결정되지 않으면 근거 부족을 고르라."),
                           "criteria": {k: criteria[k] for k in shuffled(rng, names)}},
                       target=names[NLI_MAP[target_name]], group=group, text_keys=keys,
                       basis="upstream_nli_label"))
    elif task == "nsmc":
        text = source["request"]["state"]
        out.append(row(family="requestion_review_intensity", task="flash_requestion", lang=lang,
                       state=text, qid="intensity", question={
                           "type": "score", "instructions": (
                               "Rate how positive the review is." if rng.random() < 0.5
                               else "리뷰가 얼마나 긍정적인지 평가하라."),
                           "criteria": ["0: 매우 부정", "1: 부정", "2: 중립·혼합", "3: 긍정",
                                        "4: 매우 긍정"]},
                       target=None, group=group, text_keys=keys))
        out.append(row(family="requestion_review_aspect", task="flash_requestion", lang=lang,
                       state=text, qid="aspect", question={
                           "type": "choice", "instructions": "리뷰가 주로 평가하는 대상은?",
                           "criteria": {k: v for k, v in shuffled(rng, [
                               ("연기", "배우의 연기"), ("줄거리", "이야기와 각본"),
                               ("영상", "촬영·음악·연출"), ("전반", "특정 요소 없이 전체 인상"),
                               ("기타", "영화 외적인 내용")])}},
                       target=None, group=group, text_keys=keys))
    elif task == "klue_ynat":
        candidates = source["candidate_ids"]
        target_name = source.get("target")
        if target_name in candidates:
            probe = rng.choice(candidates)
            out.append(row(family="requestion_topic_binary", task="flash_requestion", lang=lang,
                           state=source["request"]["state"], qid="is_topic", question={
                               "type": "noul", "instructions": (
                                   f"Is this headline mainly about {probe}?" if rng.random() < 0.5
                                   else f"이 뉴스 제목의 주된 분야가 '{probe}'인가?"),
                               "criteria": None},
                           target="yes" if probe == target_name else "no", group=group,
                           text_keys=keys, basis="upstream_topic_label"))
        out.append(row(family="requestion_headline_named_org", task="flash_requestion",
                       lang=lang, state=source["request"]["state"], qid="names_org", question={
                           "type": "noul", "instructions": "제목이 특정 기업이나 기관을 직접 언급하는가?",
                           "criteria": None}, target=None, group=group, text_keys=keys))
    elif task in ("massive_en-US", "massive_ko-KR", "banking77"):
        state = source["request"]["state"]
        english = lang == "en"
        out.append(row(family="requestion_utterance_urgency", task="flash_requestion", lang=lang,
                       state=state, qid="urgency", question={
                           "type": "score", "instructions": (
                               "How urgent is this request for the user?" if english
                               else "이 요청은 사용자에게 얼마나 급한가?"),
                           "criteria": (["0: not urgent", "1: soon", "2: urgent"] if english
                                        else ["0: 급하지 않음", "1: 곧", "2: 급함"])},
                       target=None, group=group, text_keys=keys))
        out.append(row(family="requestion_utterance_action", task="flash_requestion", lang=lang,
                       state=state, qid="needs_action", question={
                           "type": "noul", "instructions": (
                               "Does the user want the assistant to perform an action (not just "
                               "answer a question)?" if english else
                               "사용자가 단순 질문이 아니라 어떤 동작 수행을 원하는가?"),
                           "criteria": None}, target=None, group=group, text_keys=keys))
    elif task == "boolq":
        state = source["request"]["state"]
        out.append(row(family="requestion_passage_answerable", task="flash_requestion",
                       lang=lang, state=state, qid="answerable", question={
                           "type": "noul", "instructions": (
                               "Does the passage contain enough information to answer the "
                               "question with yes or no?"), "criteria": None},
                       target=None, group=group, text_keys=keys))
    return out


def text_of(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)
