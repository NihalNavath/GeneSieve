from client import GenesieveEnv
from models import GenesieveAction
import random


def run_episode():
    with GenesieveEnv(base_url="http://localhost:8000").sync() as env:
        res = env.reset()

        print("\n=== NEW EPISODE ===")
        print("Organism:", res.observation.organism)
        print("Budget:", res.observation.budget_remaining)

        genes = res.observation.genes_available

        step = 0

        while not res.done:
            step += 1

            # pick random gene
            gene = random.choice(genes)["gene_name"]

            # simple policy (random tools)
            tool = random.choice([
                "inspect_gene",
                "check_human_homolog",
                "test_binding",
            ])

            action = GenesieveAction(
                tool=tool,
                args={"gene_name": gene}
            )

            res = env.step(action)

            print(f"\nStep {step}")
            print("Action:", tool, gene)
            print("Reward:", res.reward)
            print("Last result:", res.observation.last_result)
            print("Done:", res.done)

            # stop early if budget low
            if res.observation.budget_remaining <= 1:
                break

        # final submission
        gene = random.choice(genes)["gene_name"]

        print("\nSubmitting:", gene)

        res = env.step(
            GenesieveAction(
                tool="submit_target",
                args={"gene_name": gene}
            )
        )

        print("\n=== FINAL ===")
        print("Final Reward:", res.reward)
        print("Done:", res.done)


if __name__ == "__main__":
    for _ in range(3):
        run_episode()