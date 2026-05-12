import numpy as np
import matplotlib.pyplot as plt

from ga_core import create_population, evolve_population


POP_SIZE = 100
CHROMOSOME_LENGTH = 100
GENERATIONS = 200
RUNS = 20


def fitness(individual):
    return sum(individual)


def run_ga(crossover_rate=0.7, mutation_rate=0.001):
    population = create_population(POP_SIZE, CHROMOSOME_LENGTH)

    best_fitness_history = []
    avg_fitness_history = []
    found_generation = None

    for generation in range(1, GENERATIONS + 1):
        fitness_values = [fitness(ind) for ind in population]

        best_fitness = max(fitness_values)
        avg_fitness = np.mean(fitness_values)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        if best_fitness == CHROMOSOME_LENGTH:
            found_generation = generation
            break

        population = evolve_population(
            population=population,
            fitness_function=fitness,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            pop_size=POP_SIZE
        )

    return {
        "found_generation": found_generation,
        "best_fitness_history": best_fitness_history,
        "avg_fitness_history": avg_fitness_history
    }


def run_experiment(crossover_rate=0.7, mutation_rate=0.001):
    found_generations = []
    successful_runs = 0
    all_best_histories = []
    all_avg_histories = []

    for _ in range(RUNS):
        result = run_ga(crossover_rate, mutation_rate)

        if result["found_generation"] is not None:
            successful_runs += 1
            found_generations.append(result["found_generation"])

        all_best_histories.append(result["best_fitness_history"])
        all_avg_histories.append(result["avg_fitness_history"])

    mean_generation = np.mean(found_generations) if found_generations else None

    return {
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "successful_runs": successful_runs,
        "mean_generation": mean_generation,
        "best_histories": all_best_histories,
        "avg_histories": all_avg_histories
    }


def pad_histories(histories):
    max_len = max(len(h) for h in histories)
    padded = []

    for h in histories:
        if len(h) < max_len:
            h = h + [h[-1]] * (max_len - len(h))
        padded.append(h)

    return np.array(padded)


def plot_experiment(result, filename):
    best = pad_histories(result["best_histories"])
    avg = pad_histories(result["avg_histories"])

    mean_best = np.mean(best, axis=0)
    mean_avg = np.mean(avg, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_best, label="Best fitness")
    plt.plot(mean_avg, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(
        f"GA Performance | Pc={result['crossover_rate']}, "
        f"Pm={result['mutation_rate']}"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    experiments = [
        (0.7, 0.001),
        (0.0, 0.001),
        (0.3, 0.001),
        (0.5, 0.001),
        (0.9, 0.001),
        (0.7, 0.0001),
        (0.7, 0.005),
        (0.7, 0.01),
    ]

    for pc, pm in experiments:
        result = run_experiment(pc, pm)

        print("-" * 50)
        print(f"Crossover rate Pc: {pc}")
        print(f"Mutation rate Pm: {pm}")
        print(f"Successful runs: {result['successful_runs']} / {RUNS}")
        print(f"Mean generation found: {result['mean_generation']}")

        filename = f"results/ga_9_1_pc_{pc}_pm_{pm}.png"
        plot_experiment(result, filename)