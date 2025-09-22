# inspect_dataset.py
from data_loader import make_dataloader

DATA_DIR = "Craftax_Baselines/craftax_classic_200M_dataset"

def main():
    loader = make_dataloader(DATA_DIR, batch_size=32, subset_fraction=0.01)

    # Take just one batch
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("obs shape:", batch["obs"].shape)
        print("action shape:", batch["action"].shape)
        print("reward stats:", batch["reward"].min().item(), batch["reward"].max().item())
        print("done % true:", batch["done"].float().mean().item())
        print("Example obs vector:", batch["obs"][0, :10])
        break  # Stop after one batch

if __name__ == "__main__":
    main()
