#!/usr/bin/env python3
"""新生儿遗传病推测工具（基于孟德尔定律与家系表型推断）。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple


DISEASE_DB = {
    "囊性纤维化": {"mode": "AR", "allele_freq": 0.02},
    "苯丙酮尿症": {"mode": "AR", "allele_freq": 0.01},
    "地中海贫血": {"mode": "AR", "allele_freq": 0.03},
    "亨廷顿舞蹈症": {"mode": "AD", "allele_freq": 0.01},
    "软骨发育不全": {"mode": "AD", "allele_freq": 0.005},
}


@dataclass(frozen=True)
class InheritanceModel:
    mode: str  # AR 或 AD
    mutant_allele_freq: float

    @property
    def genotypes(self) -> Tuple[str, str, str]:
        return ("AA", "Aa", "aa")

    def founder_prior(self, genotype: str) -> float:
        q = self.mutant_allele_freq
        if self.mode == "AR":
            table = {"AA": (1 - q) ** 2, "Aa": 2 * q * (1 - q), "aa": q**2}
        elif self.mode == "AD":
            table = {"AA": q**2, "Aa": 2 * q * (1 - q), "aa": (1 - q) ** 2}
        else:
            raise ValueError(f"不支持的遗传模式: {self.mode}")
        return table[genotype]

    def is_affected(self, genotype: str) -> bool:
        if self.mode == "AR":
            return genotype == "aa"
        if self.mode == "AD":
            return genotype in {"AA", "Aa"}
        raise ValueError(f"不支持的遗传模式: {self.mode}")

    def child_prob(self, parent1: str, parent2: str, child: str) -> float:
        p1 = gamete_probs(parent1)
        p2 = gamete_probs(parent2)
        total = 0.0
        for a1, pa1 in p1.items():
            for a2, pa2 in p2.items():
                g = normalize_genotype(a1 + a2)
                if g == child:
                    total += pa1 * pa2
        return total


def normalize_genotype(raw: str) -> str:
    if raw in {"AA", "Aa", "aA", "aa"}:
        if raw in {"Aa", "aA"}:
            return "Aa"
        return raw
    raise ValueError(f"无效基因型: {raw}")


def gamete_probs(genotype: str) -> Dict[str, float]:
    if genotype == "AA":
        return {"A": 1.0}
    if genotype == "Aa":
        return {"A": 0.5, "a": 0.5}
    if genotype == "aa":
        return {"a": 1.0}
    raise ValueError(f"无效基因型: {genotype}")


def parse_status(raw: str) -> bool:
    value = raw.strip().lower()
    affected_values = {"1", "y", "yes", "是", "患病", "有", "true", "t"}
    unaffected_values = {"0", "n", "no", "否", "未患病", "无", "false", "f"}
    if value in affected_values:
        return True
    if value in unaffected_values:
        return False
    raise ValueError("请输入 患病/未患病（或 是/否，1/0）")


def choose_model(disease_name: str) -> InheritanceModel:
    matched = DISEASE_DB.get(disease_name)
    if matched:
        return InheritanceModel(mode=matched["mode"], mutant_allele_freq=matched["allele_freq"])

    print(f"未在内置数据库中找到『{disease_name}』。")
    mode = input("请输入遗传模式（AR=常染色体隐性，AD=常染色体显性）: ").strip().upper()
    while mode not in {"AR", "AD"}:
        mode = input("输入无效，请输入 AR 或 AD: ").strip().upper()

    allele_freq = input("请输入致病等位基因频率（0~1，默认0.01）: ").strip()
    q = 0.01 if not allele_freq else float(allele_freq)
    if not (0 < q < 1):
        raise ValueError("等位基因频率必须在 0~1 之间")

    return InheritanceModel(mode=mode, mutant_allele_freq=q)


def infer_family(model: InheritanceModel, observed: Dict[str, bool]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    roles = [
        "father",
        "mother",
        "paternal_grandfather",
        "paternal_uncle",
        "maternal_aunt",
    ]

    posteriors = {r: {g: 0.0 for g in model.genotypes} for r in roles}
    baby_posterior = {g: 0.0 for g in model.genotypes}

    z = 0.0
    for pgf, pgm, mgf, mgm in product(model.genotypes, repeat=4):
        founder_weight = (
            model.founder_prior(pgf)
            * model.founder_prior(pgm)
            * model.founder_prior(mgf)
            * model.founder_prior(mgm)
        )
        if founder_weight == 0:
            continue

        for father, uncle, mother, aunt in product(model.genotypes, repeat=4):
            mendel_weight = (
                model.child_prob(pgf, pgm, father)
                * model.child_prob(pgf, pgm, uncle)
                * model.child_prob(mgf, mgm, mother)
                * model.child_prob(mgf, mgm, aunt)
            )
            if mendel_weight == 0:
                continue

            assignment = {
                "father": father,
                "mother": mother,
                "paternal_grandfather": pgf,
                "paternal_uncle": uncle,
                "maternal_aunt": aunt,
            }

            valid = True
            for role, status in observed.items():
                if model.is_affected(assignment[role]) != status:
                    valid = False
                    break
            if not valid:
                continue

            weight = founder_weight * mendel_weight
            z += weight
            for role in roles:
                posteriors[role][assignment[role]] += weight

            for baby_g in model.genotypes:
                baby_posterior[baby_g] += weight * model.child_prob(father, mother, baby_g)

    if z == 0:
        raise RuntimeError("在当前输入条件下不存在满足条件的家系组合，请检查患病状态或遗传模式。")

    for role in roles:
        for g in model.genotypes:
            posteriors[role][g] /= z
    for g in model.genotypes:
        baby_posterior[g] /= z

    return posteriors, baby_posterior


def print_distribution(title: str, dist: Dict[str, float], model: InheritanceModel) -> None:
    print(f"\n{title}")
    for g, p in sorted(dist.items(), key=lambda x: x[0]):
        tag = "患病" if model.is_affected(g) else "未患病"
        print(f"  {g}: {p * 100:6.2f}% ({tag})")


def main() -> None:
    print("=== 新生儿遗传病推测工具（孟德尔杂交）===")
    disease = input("请输入疾病名称: ").strip()
    model = choose_model(disease)

    prompts = {
        "father": "父亲是否患病? ",
        "mother": "母亲是否患病? ",
        "paternal_grandfather": "爷爷（父系祖父）是否患病? ",
        "paternal_uncle": "叔叔（父亲的兄弟）是否患病? ",
        "maternal_aunt": "姨姨（母亲的姐妹）是否患病? ",
    }

    observed = {}
    for role, prompt in prompts.items():
        while True:
            try:
                observed[role] = parse_status(input(prompt))
                break
            except ValueError as err:
                print(err)

    posteriors, baby = infer_family(model, observed)

    print(f"\n疾病: {disease} | 遗传模式: {model.mode}")
    for role, label in [
        ("paternal_grandfather", "爷爷"),
        ("father", "父亲"),
        ("paternal_uncle", "叔叔"),
        ("mother", "母亲"),
        ("maternal_aunt", "姨姨"),
    ]:
        print_distribution(f"{label}基因型后验概率", posteriors[role], model)

    print_distribution("新生儿基因型概率", baby, model)


if __name__ == "__main__":
    main()
