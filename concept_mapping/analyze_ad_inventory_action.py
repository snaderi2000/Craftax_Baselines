import argparse
import csv
import json
import os

import torch as th


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an Achievement-Distillation transition dataset for whether an "
            "inventory item appears after a specific action."
        )
    )
    parser.add_argument(
        "--dataset",
        default="/home/shawheen/Achievement-Distillation/affordance_transitions.pt",
        help="Path produced by Achievement-Distillation/collect_affordance_transitions.py",
    )
    parser.add_argument("--action", default="make_wood_pickaxe")
    parser.add_argument("--item", default="wood_pickaxe")
    parser.add_argument("--consumed_item", default="wood")
    parser.add_argument("--achievement", default="make_wood_pickaxe")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    data = th.load(args.dataset, map_location="cpu")
    meta = data.get("metadata", {})
    action_names = meta.get("action_names")
    inventory_keys = meta.get("inventory_keys")
    achievement_keys = meta.get("achievement_keys", [])
    if not action_names or not inventory_keys:
        raise ValueError("Dataset metadata must include action_names and inventory_keys.")
    for value, values, name in [
        (args.action, action_names, "action"),
        (args.item, inventory_keys, "item"),
        (args.consumed_item, inventory_keys, "consumed_item"),
    ]:
        if value not in values:
            raise ValueError(f"{name}={value!r} not found. Available: {values}")

    action_id = action_names.index(args.action)
    item_id = inventory_keys.index(args.item)
    consumed_id = inventory_keys.index(args.consumed_item)
    achievement_id = achievement_keys.index(args.achievement) if args.achievement in achievement_keys else None

    actions = data["actions"].long()
    inventories = data["inventories"].float()
    next_inventories = data["next_inventories"].float()
    rewards = data.get("rewards", th.zeros_like(actions, dtype=th.float32)).float()
    dones = data.get("dones", th.zeros_like(actions, dtype=th.bool)).bool()

    action_mask = actions == action_id
    item_delta = next_inventories[:, item_id] - inventories[:, item_id]
    consumed_delta = next_inventories[:, consumed_id] - inventories[:, consumed_id]
    appears = action_mask & (item_delta > 0.5)
    appears_and_consumes = appears & (consumed_delta < -0.5)
    positive = action_mask & (rewards > 1e-6)
    failed_no_item = action_mask & ~appears

    if achievement_id is not None and "achievements" in data and "next_achievements" in data:
        achievements = data["achievements"].float()
        next_achievements = data["next_achievements"].float()
        achievement_delta = next_achievements[:, achievement_id] - achievements[:, achievement_id]
        achievement_new = action_mask & (achievement_delta > 0.5)
    else:
        achievement_new = th.zeros_like(action_mask)

    attempts = int(action_mask.sum())
    rows = [
        ("dataset", args.dataset, None),
        ("checkpoint", str(meta.get("checkpoint", "")), None),
        ("action", args.action, action_id),
        ("attempts", attempts, _pct(attempts, len(actions))),
        (f"new_{args.item}", int(appears.sum()), _pct(int(appears.sum()), attempts)),
        (
            f"new_{args.item}_and_{args.consumed_item}_decreases",
            int(appears_and_consumes.sum()),
            _pct(int(appears_and_consumes.sum()), attempts),
        ),
        (
            f"achievement_{args.achievement}_delta",
            int(achievement_new.sum()),
            _pct(int(achievement_new.sum()), attempts),
        ),
        ("positive_reward", int(positive.sum()), _pct(int(positive.sum()), attempts)),
        (f"failed_no_new_{args.item}", int(failed_no_item.sum()), _pct(int(failed_no_item.sum()), attempts)),
        ("new_item_and_positive", int((appears & positive).sum()), _pct(int((appears & positive).sum()), attempts)),
        ("positive_without_new_item", int((positive & ~appears).sum()), _pct(int((positive & ~appears).sum()), attempts)),
        ("new_item_without_positive", int((appears & ~positive).sum()), _pct(int((appears & ~positive).sum()), attempts)),
        ("done_among_new_item", int(dones[appears].sum()), _pct(int(dones[appears].sum()), int(appears.sum()))),
    ]
    if int(appears.sum()):
        rows.append(("mean_reward_new_item", float(rewards[appears].mean()), None))
    if int(failed_no_item.sum()):
        rows.append(("mean_reward_failed_no_item", float(rewards[failed_no_item].mean()), None))

    print("metric,value,percent")
    for metric, value, percent in rows:
        pct_text = "" if percent is None else f"{percent:.4f}"
        print(f"{metric},{value},{pct_text}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value", "percent"])
            writer.writerows(rows)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            metric: {"value": value, "percent": percent}
            for metric, value, percent in rows
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
