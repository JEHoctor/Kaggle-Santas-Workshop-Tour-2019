from santaspkg.cost_function import cost_function
from santaspkg.dataset import N_FAMILIES
from santaspkg.greedy import greedy
from santaspkg.mk_submit import mk_submit
from santaspkg.refinement import refine_until_convergence

import numpy as np
import multiprocessing as mp


class Chromosome(object):
    def __init__(self, codons=None):
        if codons is None:
            self.codons = np.random.randint(0, 255, size=2*N_FAMILIES, dtype=np.uint8)
        else:
            self.codons = codons
        self.codons.flags.writeable = False
        self.score = None
        self.get_score()

    def get_permutation(self):
        data_as_uint16 = np.frombuffer(self.codons.data, dtype=np.uint16)
        return np.argsort(data_as_uint16)

    @staticmethod
    def advance_index(index, max_index, alpha):
        return min(max_index, index + np.random.geometric(alpha))

    @staticmethod
    def make_mask(alpha):
        max_index = 2*N_FAMILIES
        mask = np.zeros(shape=max_index, dtype=np.bool)

        start_index = Chromosome.advance_index(0, max_index, alpha)
        end_index = Chromosome.advance_index(start_index, max_index, alpha)
        while start_index < max_index:
            mask[start_index:end_index] = True
            start_index = Chromosome.advance_index(end_index, max_index, alpha)
            end_index = Chromosome.advance_index(start_index, max_index, alpha)

        if np.random.binomial(n=1, p=0.5): # coin flip
            mask = ~mask
        return mask

    def recombine(self, other, alpha=0.005):
        mask = self.make_mask(alpha)
        return Chromosome(self.codons*mask + other.codons*(~mask))

    def get_score(self):
        if self.score is not None:
            return self.score
        else:
            initial_assignment = greedy(self.get_permutation())
            refined_assignment = refine_until_convergence(initial_assignment)
            score = cost_function(refined_assignment)
            self.score = score
            self.assignment = refined_assignment
            return self.score


def combine(chromosome1, chromosome2):
    return chromosome1.recombine(chromosome2)


def pick_parents(gene_pool, new_genes, m, n):
    pool_pvals = np.array([1/c.get_score() for c in gene_pool])
    pool_pvals /= pool_pvals.sum()
    def get_parent_from_pool():
        return gene_pool[np.random.multinomial(1, pool_pvals).argmax()]

    parents = []
    for new_gene in new_genes:
        for i in range(m):
            parents.append((new_gene, get_parent_from_pool()))
    for i in range(n):
        parents.append((get_parent_from_pool(), get_parent_from_pool()))
    return parents


def genetic_algorithm(generations=3):
    n_cpus = mp.cpu_count() - 1
    worker_pool = mp.Pool(processes=n_cpus, maxtasksperchild=100)

    gene_pool = worker_pool.starmap(Chromosome, (() for x in range(4*n_cpus)))
    gene_pool.sort(key=lambda c: c.get_score())
    for i in range(generations):
        new_genes = worker_pool.starmap(Chromosome, (() for x in range(n_cpus)))
        parents = pick_parents(gene_pool, new_genes, m=2, n=5*n_cpus)
        children = worker_pool.starmap(combine, parents)

        gene_pool += new_genes + children
        gene_pool.sort(key=lambda c: c.get_score())
        gene_pool = gene_pool[:150]

        print('current best', gene_pool[0].get_score())
        mk_submit(gene_pool[0].assignment)


if __name__ == '__main__':
    genetic_algorithm()
