import re
import json
import numpy as np
import pandas as pd

# import sys
# sys.path.append('/root/ai2/metaeval/olmo-repos/oe-eval-internal')
# from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
# primary_metrics = {}
# for task in TASK_REGISTRY.keys():
#     task_config = TASK_REGISTRY[task].__dict__.get('TASK_CONFIG_DEFAULTS', {})
#     primary_metric = task_config.get("primary_metric", None)
#     primary_metrics[task] = primary_metric
# raise RuntimeError(primary_metrics)


PRIMARY_METRICS_OLMES = {
    # Custom entries
    'autobencher': 'acc_per_char',
    'autobencher:mc': 'acc_per_char',

    'aime': 'exact_match_flex', 
    'alpaca_eval': 'win_rate', 
    'arc_challenge': 'acc_uncond', 
    'arc_challenge:mc': 'acc_raw', 
    'arc_easy': 'acc_per_char', 
    'arc_easy:mc': 'acc_raw', 
    'autobencher': 'logits_per_byte', 
    'autobencher:mc': 'acc_raw', 
    'autobencher_math': 'exact_match', 
    'bigcodebench': 'pass_at_1', 
    'bigcodebench_hard': 'pass_at_1', 
    'boolq': 'acc_raw', 
    'boolq:mc': None, 'custom_loss_sky_t1': 'bits_per_byte', 
    'custom_loss_numia_math': 'bits_per_byte', 
    'custom_loss_tulu_if': 'bits_per_byte', 
    'codex_humaneval': 'pass_at_1', 
    'codex_humanevalplus': 'pass_at_1', 
    'copa': 'acc_raw', 
    'copycolors': 'acc_uncond', 
    'copycolors:mc': 'acc_raw', 
    'coqa': 'f1', 
    'cosmosqa': 'acc_per_char', 
    'cosmosqa:mc': 'acc_raw', 
    'csqa': 'acc_uncond', 
    'csqa:mc': 'acc_raw', 
    'drop': 'f1', 
    'gsm8k': 'exact_match', 
    'gsm8k_selfc': 'maj_at_10', 
    'gsm_plus': 'exact_match', 
    'gsm_plus_selfc': None, 'gsm_symbolic_main': 'exact_match', 
    'gsm_symbolic_p1': 'exact_match', 
    'gsm_symbolic_p2': 'exact_match', 
    'gpqa': 'exact_match', 
    'hellaswag': 'acc_per_char', 
    'hellaswag:mc': 'acc_raw', 
    'ifeval': 'inst_level_loose_acc', 
    'jeopardy': 'f1', 
    'logiqa': 'acc_per_char', 
    'logiqa:mc': 'acc_raw', 
    'minerva_math_algebra': None, 'minerva_math_counting_and_probability': None, 'minerva_math_geometry': None, 'minerva_math_intermediate_algebra': None, 'minerva_math_number_theory': None, 'minerva_math_prealgebra': None, 'minerva_math_precalculus': None, 'minerva_math_500': 'exact_match', 
    'mbpp': 'pass_at_1', 
    'mbppplus': 'pass_at_1', 
    'medmcqa': 'acc_per_char', 
    'medmcqa:mc': 'acc_per_char', 
    'mmlu_abstract_algebra:mc': None, 'mmlu_abstract_algebra': None, 'mmlu_abstract_algebra:cot': 'exact_match', 
    'mmlu_anatomy:mc': None, 'mmlu_anatomy': None, 'mmlu_anatomy:cot': 'exact_match', 
    'mmlu_astronomy:mc': None, 'mmlu_astronomy': None, 'mmlu_astronomy:cot': 'exact_match', 
    'mmlu_business_ethics:mc': None, 'mmlu_business_ethics': None, 'mmlu_business_ethics:cot': 'exact_match', 
    'mmlu_clinical_knowledge:mc': None, 'mmlu_clinical_knowledge': None, 'mmlu_clinical_knowledge:cot': 'exact_match', 
    'mmlu_college_biology:mc': None, 'mmlu_college_biology': None, 'mmlu_college_biology:cot': 'exact_match', 
    'mmlu_college_chemistry:mc': None, 'mmlu_college_chemistry': None, 'mmlu_college_chemistry:cot': 'exact_match', 
    'mmlu_college_computer_science:mc': None, 'mmlu_college_computer_science': None, 'mmlu_college_computer_science:cot': 'exact_match', 
    'mmlu_college_mathematics:mc': None, 'mmlu_college_mathematics': None, 'mmlu_college_mathematics:cot': 'exact_match', 
    'mmlu_college_medicine:mc': None, 'mmlu_college_medicine': None, 'mmlu_college_medicine:cot': 'exact_match', 
    'mmlu_college_physics:mc': None, 'mmlu_college_physics': None, 'mmlu_college_physics:cot': 'exact_match', 
    'mmlu_computer_security:mc': None, 'mmlu_computer_security': None, 'mmlu_computer_security:cot': 'exact_match', 
    'mmlu_conceptual_physics:mc': None, 'mmlu_conceptual_physics': None, 'mmlu_conceptual_physics:cot': 'exact_match', 
    'mmlu_econometrics:mc': None, 'mmlu_econometrics': None, 'mmlu_econometrics:cot': 'exact_match', 
    'mmlu_electrical_engineering:mc': None, 'mmlu_electrical_engineering': None, 'mmlu_electrical_engineering:cot': 'exact_match', 
    'mmlu_elementary_mathematics:mc': None, 'mmlu_elementary_mathematics': None, 'mmlu_elementary_mathematics:cot': 'exact_match', 
    'mmlu_formal_logic:mc': None, 'mmlu_formal_logic': None, 'mmlu_formal_logic:cot': 'exact_match', 
    'mmlu_global_facts:mc': None, 'mmlu_global_facts': None, 'mmlu_global_facts:cot': 'exact_match', 
    'mmlu_high_school_biology:mc': None, 'mmlu_high_school_biology': None, 'mmlu_high_school_biology:cot': 'exact_match', 
    'mmlu_high_school_chemistry:mc': None, 'mmlu_high_school_chemistry': None, 'mmlu_high_school_chemistry:cot': 'exact_match', 
    'mmlu_high_school_computer_science:mc': None, 'mmlu_high_school_computer_science': None, 'mmlu_high_school_computer_science:cot': 'exact_match', 
    'mmlu_high_school_european_history:mc': None, 'mmlu_high_school_european_history': None, 'mmlu_high_school_european_history:cot': 'exact_match', 
    'mmlu_high_school_geography:mc': None, 'mmlu_high_school_geography': None, 'mmlu_high_school_geography:cot': 'exact_match', 
    'mmlu_high_school_government_and_politics:mc': None, 'mmlu_high_school_government_and_politics': None, 'mmlu_high_school_government_and_politics:cot': 'exact_match', 
    'mmlu_high_school_macroeconomics:mc': None, 'mmlu_high_school_macroeconomics': None, 'mmlu_high_school_macroeconomics:cot': 'exact_match', 
    'mmlu_high_school_mathematics:mc': None, 'mmlu_high_school_mathematics': None, 'mmlu_high_school_mathematics:cot': 'exact_match', 
    'mmlu_high_school_microeconomics:mc': None, 'mmlu_high_school_microeconomics': None, 'mmlu_high_school_microeconomics:cot': 'exact_match', 
    'mmlu_high_school_physics:mc': None, 'mmlu_high_school_physics': None, 'mmlu_high_school_physics:cot': 'exact_match', 
    'mmlu_high_school_psychology:mc': None, 'mmlu_high_school_psychology': None, 'mmlu_high_school_psychology:cot': 'exact_match', 
    'mmlu_high_school_statistics:mc': None, 'mmlu_high_school_statistics': None, 'mmlu_high_school_statistics:cot': 'exact_match', 
    'mmlu_high_school_us_history:mc': None, 'mmlu_high_school_us_history': None, 'mmlu_high_school_us_history:cot': 'exact_match', 
    'mmlu_high_school_world_history:mc': None, 'mmlu_high_school_world_history': None, 'mmlu_high_school_world_history:cot': 'exact_match', 
    'mmlu_human_aging:mc': None, 'mmlu_human_aging': None, 'mmlu_human_aging:cot': 'exact_match', 
    'mmlu_human_sexuality:mc': None, 'mmlu_human_sexuality': None, 'mmlu_human_sexuality:cot': 'exact_match', 
    'mmlu_international_law:mc': None, 'mmlu_international_law': None, 'mmlu_international_law:cot': 'exact_match', 
    'mmlu_jurisprudence:mc': None, 'mmlu_jurisprudence': None, 'mmlu_jurisprudence:cot': 'exact_match', 
    'mmlu_logical_fallacies:mc': None, 'mmlu_logical_fallacies': None, 'mmlu_logical_fallacies:cot': 'exact_match', 
    'mmlu_machine_learning:mc': None, 'mmlu_machine_learning': None, 'mmlu_machine_learning:cot': 'exact_match', 
    'mmlu_management:mc': None, 'mmlu_management': None, 'mmlu_management:cot': 'exact_match', 
    'mmlu_marketing:mc': None, 'mmlu_marketing': None, 'mmlu_marketing:cot': 'exact_match', 
    'mmlu_medical_genetics:mc': None, 'mmlu_medical_genetics': None, 'mmlu_medical_genetics:cot': 'exact_match', 
    'mmlu_miscellaneous:mc': None, 'mmlu_miscellaneous': None, 'mmlu_miscellaneous:cot': 'exact_match', 
    'mmlu_moral_disputes:mc': None, 'mmlu_moral_disputes': None, 'mmlu_moral_disputes:cot': 'exact_match', 
    'mmlu_moral_scenarios:mc': None, 'mmlu_moral_scenarios': None, 'mmlu_moral_scenarios:cot': 'exact_match', 
    'mmlu_nutrition:mc': None, 'mmlu_nutrition': None, 'mmlu_nutrition:cot': 'exact_match', 
    'mmlu_philosophy:mc': None, 'mmlu_philosophy': None, 'mmlu_philosophy:cot': 'exact_match', 
    'mmlu_prehistory:mc': None, 'mmlu_prehistory': None, 'mmlu_prehistory:cot': 'exact_match', 
    'mmlu_professional_accounting:mc': None, 'mmlu_professional_accounting': None, 'mmlu_professional_accounting:cot': 'exact_match', 
    'mmlu_professional_law:mc': None, 'mmlu_professional_law': None, 'mmlu_professional_law:cot': 'exact_match', 
    'mmlu_professional_medicine:mc': None, 'mmlu_professional_medicine': None, 'mmlu_professional_medicine:cot': 'exact_match', 
    'mmlu_professional_psychology:mc': None, 'mmlu_professional_psychology': None, 'mmlu_professional_psychology:cot': 'exact_match', 
    'mmlu_public_relations:mc': None, 'mmlu_public_relations': None, 'mmlu_public_relations:cot': 'exact_match', 
    'mmlu_security_studies:mc': None, 'mmlu_security_studies': None, 'mmlu_security_studies:cot': 'exact_match', 
    'mmlu_sociology:mc': None, 'mmlu_sociology': None, 'mmlu_sociology:cot': 'exact_match', 
    'mmlu_us_foreign_policy:mc': None, 'mmlu_us_foreign_policy': None, 'mmlu_us_foreign_policy:cot': 'exact_match', 
    'mmlu_virology:mc': None, 'mmlu_virology': None, 'mmlu_virology:cot': 'exact_match', 
    'mmlu_world_religions:mc': None, 'mmlu_world_religions': None, 'mmlu_world_religions:cot': 'exact_match', 
    'mmlu_pro_math:cot': 'exact_match', 
    'mmlu_pro_health:cot': 'exact_match', 
    'mmlu_pro_physics:cot': 'exact_match', 
    'mmlu_pro_business:cot': 'exact_match', 
    'mmlu_pro_biology:cot': 'exact_match', 
    'mmlu_pro_chemistry:cot': 'exact_match', 
    'mmlu_pro_computer science:cot': 'exact_match', 
    'mmlu_pro_economics:cot': 'exact_match', 
    'mmlu_pro_engineering:cot': 'exact_match', 
    'mmlu_pro_philosophy:cot': 'exact_match', 
    'mmlu_pro_other:cot': 'exact_match', 
    'mmlu_pro_history:cot': 'exact_match', 
    'mmlu_pro_psychology:cot': 'exact_match', 
    'mmlu_pro_law:cot': 'exact_match', 
    'mmlu_pro_math': None, 'mmlu_pro_health': None, 'mmlu_pro_physics': None, 'mmlu_pro_business': None, 'mmlu_pro_biology': None, 'mmlu_pro_chemistry': None, 'mmlu_pro_computer science': None, 'mmlu_pro_economics': None, 'mmlu_pro_engineering': None, 'mmlu_pro_philosophy': None, 'mmlu_pro_other': None, 'mmlu_pro_history': None, 'mmlu_pro_psychology': None, 'mmlu_pro_law': None, 'mmlu_pro_math:rc': None, 'mmlu_pro_health:rc': None, 'mmlu_pro_physics:rc': None, 'mmlu_pro_business:rc': None, 'mmlu_pro_biology:rc': None, 'mmlu_pro_chemistry:rc': None, 'mmlu_pro_computer science:rc': None, 'mmlu_pro_economics:rc': None, 'mmlu_pro_engineering:rc': None, 'mmlu_pro_philosophy:rc': None, 'mmlu_pro_other:rc': None, 'mmlu_pro_history:rc': None, 'mmlu_pro_psychology:rc': None, 'mmlu_pro_law:rc': None, 'mt_eval_refinement_single': 'llm_score', 
    'mt_eval_refinement_multi': 'llm_score', 
    'mt_eval_expansion_single': 'llm_score', 
    'mt_eval_expansion_multi': 'llm_score', 
    'mt_eval_follow-up_single': 'llm_score', 
    'mt_eval_follow-up_multi': 'llm_score', 
    'mt_eval_recollection_single_cls': 'llm_score', 
    'mt_eval_recollection_multi_cls': 'llm_score', 
    'mt_eval_recollection_single_global-inst': 'llm_score', 
    'mt_eval_recollection_multi_global-inst': 'llm_score', 
    'naturalqs_open': 'f1', 
    'openbookqa': 'acc_uncond', 
    'openbookqa:mc': 'acc_raw', 
    'paloma_4chan_meta_sep': None, 'paloma_c4_100_domains': None, 'paloma_c4_en': None, 'paloma_dolma_100_programing_languages': None, 'paloma_dolma_100_subreddits': None, 'paloma_dolma-v1_5': None, 'paloma_falcon-refinedweb': None, 'paloma_gab': None, 'paloma_m2d2_s2orc_unsplit': None, 'paloma_m2d2_wikipedia_unsplit': None, 'paloma_manosphere_meta_sep': None, 'paloma_mc4': None, 'paloma_ptb': None, 'paloma_redpajama': None, 'paloma_twitterAAE_HELM_fixed': None, 'paloma_wikitext_103': None, 'llm_compression_arxiv_math': None, 'llm_compression_cc': None, 'llm_compression_python': None, 'piqa': 'acc_per_char', 
    'piqa:mc': 'acc_raw', 
    'popqa': 'exact_match', 
    'sciq': 'acc_raw', 
    'socialiqa': 'acc_per_char', 
    'socialiqa:mc': 'acc_raw', 
    'squad': 'f1', 
    'squad2': 'f1', 
    'triviaqa': 'f1', 
    'truthfulqa': 'mc1', 
    'tydiqa_english': None, 'tydiqa_arabic': None, 'tydiqa_bengali': None, 'tydiqa_finnish': None, 'tydiqa_indonesian': None, 'tydiqa_korean': None, 'tydiqa_russian': None, 'tydiqa_swahili': None, 'tydiqa_telugu': None, 'winogrande': 'acc_raw', 
    'winogrande:mc': 'acc_raw', 
    'zero_scrolls_gov_report': 'rougeL_f1', 
    'zero_scrolls_summ_screen_fd': 'rougeL_f1', 
    'zero_scrolls_qmsum': 'rougeL_f1', 
    'zero_scrolls_qasper': 'f1', 
    'zero_scrolls_narrative_qa': 'f1', 
    'zero_scrolls_quality': 'exact_match', 
    'arc_challenge:para': 'acc_per_char', 
    'arc_easy:para': 'acc_per_char', 
    'boolq:para': 'acc_raw', 
    'csqa:para': 'acc_uncond', 
    'hellaswag:para': 'acc_per_char', 
    'openbookqa:para': 'acc_uncond', 
    'piqa:para': 'acc_per_char', 
    'socialiqa:para': 'acc_per_char', 
    'winogrande:para': 'acc_raw', 
    'mmlu_abstract_algebra:para': 'acc_per_char', 
    'mmlu_anatomy:para': 'acc_per_char', 
    'mmlu_astronomy:para': 'acc_per_char', 
    'mmlu_business_ethics:para': 'acc_per_char', 
    'mmlu_clinical_knowledge:para': 'acc_per_char', 
    'mmlu_college_biology:para': 'acc_per_char', 
    'mmlu_college_chemistry:para': 'acc_per_char', 
    'mmlu_college_computer_science:para': 'acc_per_char', 
    'mmlu_college_mathematics:para': 'acc_per_char', 
    'mmlu_college_medicine:para': 'acc_per_char', 
    'mmlu_college_physics:para': 'acc_per_char', 
    'mmlu_computer_security:para': 'acc_per_char', 
    'mmlu_conceptual_physics:para': 'acc_per_char', 
    'mmlu_econometrics:para': 'acc_per_char', 
    'mmlu_electrical_engineering:para': 'acc_per_char', 
    'mmlu_elementary_mathematics:para': 'acc_per_char', 
    'mmlu_formal_logic:para': 'acc_per_char', 
    'mmlu_global_facts:para': 'acc_per_char', 
    'mmlu_high_school_biology:para': 'acc_per_char', 
    'mmlu_high_school_chemistry:para': 'acc_per_char', 
    'mmlu_high_school_computer_science:para': 'acc_per_char', 
    'mmlu_high_school_european_history:para': 'acc_per_char', 
    'mmlu_high_school_geography:para': 'acc_per_char', 
    'mmlu_high_school_government_and_politics:para': 'acc_per_char', 
    'mmlu_high_school_macroeconomics:para': 'acc_per_char', 
    'mmlu_high_school_mathematics:para': 'acc_per_char', 
    'mmlu_high_school_microeconomics:para': 'acc_per_char', 
    'mmlu_high_school_physics:para': 'acc_per_char', 
    'mmlu_high_school_psychology:para': 'acc_per_char', 
    'mmlu_high_school_statistics:para': 'acc_per_char', 
    'mmlu_high_school_us_history:para': 'acc_per_char', 
    'mmlu_high_school_world_history:para': 'acc_per_char', 
    'mmlu_human_aging:para': 'acc_per_char', 
    'mmlu_human_sexuality:para': 'acc_per_char', 
    'mmlu_international_law:para': 'acc_per_char', 
    'mmlu_jurisprudence:para': 'acc_per_char', 
    'mmlu_logical_fallacies:para': 'acc_per_char', 
    'mmlu_machine_learning:para': 'acc_per_char', 
    'mmlu_management:para': 'acc_per_char', 
    'mmlu_marketing:para': 'acc_per_char', 
    'mmlu_medical_genetics:para': 'acc_per_char', 
    'mmlu_miscellaneous:para': 'acc_per_char', 
    'mmlu_moral_disputes:para': 'acc_per_char', 
    'mmlu_moral_scenarios:para': 'acc_per_char', 
    'mmlu_nutrition:para': 'acc_per_char', 
    'mmlu_philosophy:para': 'acc_per_char', 
    'mmlu_prehistory:para': 'acc_per_char', 
    'mmlu_professional_accounting:para': 'acc_per_char', 
    'mmlu_professional_law:para': 'acc_per_char', 
    'mmlu_professional_medicine:para': 'acc_per_char', 
    'mmlu_professional_psychology:para': 'acc_per_char', 
    'mmlu_public_relations:para': 'acc_per_char', 
    'mmlu_security_studies:para': 'acc_per_char', 
    'mmlu_sociology:para': 'acc_per_char', 
    'mmlu_us_foreign_policy:para': 'acc_per_char', 
    'mmlu_virology:para': 'acc_per_char', 
    'mmlu_world_religions:para': 'acc_per_char', 
    'minerva_math_geometry:perturb_cot': 'acc_per_char', 
    'gsm8k:perturb_cot': 'acc_per_char', 
    'minerva_math_intermediate_algebra:perturb_cot': 'acc_per_char', 
    'minerva_math_number_theory:perturb_cot': 'acc_per_char', 
    'minerva_math_algebra:perturb_cot': 'acc_per_char', 
    'minerva_math_prealgebra:perturb_cot': 'acc_per_char', 
    'minerva_math_counting_and_probability:perturb_cot': 'acc_per_char', 
    'arc_challenge:enlarge': 'acc_uncond', 
    'arc_easy:enlarge': 'acc_per_char', 
    'boolq:enlarge': 'acc_raw', 
    'csqa:enlarge': 'acc_uncond', 
    'hellaswag:enlarge': 'acc_per_char', 
    'openbookqa:enlarge': 'acc_uncond', 
    'piqa:enlarge': 'acc_per_char', 
    'socialiqa:enlarge': 'acc_per_char', 
    'arc_challenge:distractors': 'acc_uncond', 
    'arc_easy:distractors': 'acc_per_char', 
    'boolq:distractors': 'acc_raw', 
    'csqa:distractors': 'acc_uncond', 
    'hellaswag:distractors': 'acc_per_char', 
    'openbookqa:distractors': 'acc_uncond', 
    'piqa:distractors': 'acc_per_char', 
    'socialiqa:distractors': 'acc_per_char', 
    'drop:perturb_rc': 'acc_per_char', 
    'gsm8k:perturb_rc': 'acc_per_char', 
    'jeopardy:perturb_rc': 'acc_per_char', 
    'naturalqs:perturb_rc': 'acc_per_char', 
    'squad:perturb_rc': 'acc_per_char', 
    'triviaqa:perturb_rc': 'acc_per_char'
}


generate_primary_metrics = [
    "em",
    "f1",
    "exact_match",
    "pass_at_1",
    # "prompt_level_loose_acc",
    # "maj_at_1"
]


def get_tasks(file_path):
    with open(file_path, "r") as f:
        tasks = [t.split(":")[0] for t in f.readlines()]
    return tasks


def safe_eval(x):
    """Utility for reading 'metrics' col which is a dict in DataFrame"""
    try:
        result = eval(x)
        # Traverse dict to replace NaN values
        if isinstance(result, dict):
            result = {
                key: (None if (isinstance(value, float) and np.isnan(value)) else value)
                for key, value in result.items()
            }
        return result
    except:
        # If fails, return the original string or handle it as needed
        return x


def log_sum_exp(log_probs):
    """Numerical stable way to compute log(sum(exp(log_probs)))"""
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))


def check_finite_and_nan(value, name):
    assert np.isfinite(value), f"{name}: {value} is inf or -inf"
    assert not np.isnan(value), f"{name}: {value} is NaN"


def process_predictions_cheap_decisions(prediction):
    metrics = prediction["metrics"]
    model_outputs = prediction["model_output"]

    # 1. RC tasks
    if all(key in metrics for key in ["acc_raw", "acc_per_char"]):
        correct_idx = metrics["correct_choice"]
        correct_output = model_outputs[correct_idx]

        # Compute correct seq
        correct_logit = correct_output["sum_logits"]
        correct_logit_per_token = correct_output["logits_per_token"]
        correct_logit_per_char = correct_output["logits_per_char"]
        correct_logit_per_byte = correct_output["logits_per_byte"]

        # Compute margin
        correct_prob = np.exp(correct_logit)
        correct_prob_per_token = np.exp(correct_logit_per_token)
        correct_prob_per_char = np.exp(correct_logit_per_char)
        correct_prob_per_byte = np.exp(correct_logit_per_byte)
        incorrect_probs = [
            np.exp(out["sum_logits"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_token = [
            np.exp(out["logits_per_token"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_char = [
            np.exp(out["logits_per_char"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_byte = [
            np.exp(out["logits_per_byte"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]

        # Compute uncond
        if all("sum_logits_uncond" in option for option in model_outputs):
            uncond_logits = np.array(
                [option["sum_logits_uncond"] for option in model_outputs]
            )
            uncond_correct_logit = uncond_logits[correct_idx]
            uncond_correct_prob = np.exp(uncond_correct_logit)
            uncond_correct_prob_per_token = np.exp(
                uncond_correct_logit / correct_output["num_tokens"]
            )
            uncond_correct_prob_per_char = np.exp(
                uncond_correct_logit / correct_output["num_chars"]
            )
            uncond_correct_prob_per_byte = np.exp(
                uncond_correct_logit / correct_output["num_bytes"]
            )
            # sum
            uncond_total_logit = log_sum_exp(uncond_logits)
            uncond_total_prob = np.exp(uncond_total_logit)
        else:
            uncond_correct_prob = None
            uncond_total_prob = None
            uncond_correct_prob_per_token = None
            uncond_correct_prob_per_char = None
            uncond_correct_prob_per_byte = None

        if incorrect_probs and not np.isnan(correct_prob - np.max(incorrect_probs)):
            margin = correct_prob - np.max(incorrect_probs)
            margin_per_token = correct_prob_per_token - np.max(
                incorrect_probs_per_token
            )
            margin_per_char = correct_prob_per_char - np.max(incorrect_probs_per_char)
            margin_per_byte = correct_prob_per_byte - np.max(incorrect_probs_per_byte)
            assert -1 <= margin <= 1, f"Margin out of bounds: {margin}"
            assert (
                -1 <= margin_per_token <= 1
            ), f"Margin per token out of bounds: {margin_per_token}"
            assert (
                -1 <= margin_per_char <= 1
            ), f"Margin per char out of bounds: {margin_per_char}"
        else:
            margin = None
            margin_per_token = None
            margin_per_char = None
            margin_per_byte = None

        # Compute total_logit and total_prob using log-sum-exp trick
        logits = np.array([option["sum_logits"] for option in model_outputs])
        total_logit = log_sum_exp(logits)
        total_prob = np.exp(total_logit)

        logits_per_token = np.array(
            [option["logits_per_token"] for option in model_outputs]
        )
        total_logit_per_token = log_sum_exp(logits_per_token)
        total_prob_per_token = np.exp(total_logit_per_token)

        logits_per_char = np.array(
            [option["logits_per_char"] for option in model_outputs]
        )
        total_logit_per_char = log_sum_exp(logits_per_char)
        total_prob_per_char = np.exp(total_logit_per_char)

        logits_per_byte = np.array(
            [option["logits_per_char"] for option in model_outputs]
        )
        total_logit_per_byte = log_sum_exp(logits_per_byte)
        total_prob_per_byte = np.exp(total_logit_per_byte)

        norm_correct_prob = np.exp(correct_logit - total_logit)
        norm_correct_prob_per_token = np.exp(
            correct_logit_per_token - total_logit_per_token
        )
        norm_correct_prob_per_char = np.exp(
            correct_logit_per_char - total_logit_per_char
        )
        norm_correct_prob_per_byte = np.exp(
            correct_logit_per_byte - total_logit_per_byte
        )

        if not np.isnan(total_prob):
            assert (
                0 <= total_prob <= len(model_outputs)
            ), f"Total probability out of bounds ({len(model_outputs)}): {total_prob}"
            assert (
                0 <= norm_correct_prob <= 1
            ), f"Normalized correct probability out of bounds: {norm_correct_prob}"
            assert (
                0 <= norm_correct_prob_per_token <= 1
            ), f"Normalized correct probability per token out of bounds: {norm_correct_prob_per_token}"
            assert (
                0 <= norm_correct_prob_per_char <= 1
            ), f"Normalized correct probability per char out of bounds: {norm_correct_prob_per_char}"

            # Checks for inf, -inf, and NaNs
            check_finite_and_nan(total_prob, "total_prob")
            check_finite_and_nan(total_prob_per_token, "total_prob_per_token")
            check_finite_and_nan(total_prob_per_char, "total_prob_per_char")
            check_finite_and_nan(norm_correct_prob, "norm_correct_prob")
            check_finite_and_nan(
                norm_correct_prob_per_token, "norm_correct_prob_per_token"
            )
            check_finite_and_nan(
                norm_correct_prob_per_char, "norm_correct_prob_per_char"
            )

        row_dict = {
            "correct_logit": correct_logit,
            "correct_logit_per_token": correct_logit_per_token,
            "correct_logit_per_char": correct_logit_per_char,
            "correct_logit_per_byte": correct_logit_per_byte,
            "correct_prob": correct_prob,
            "correct_prob_per_token": correct_prob_per_token,
            "correct_prob_per_char": correct_prob_per_char,
            "correct_prob_per_byte": correct_prob_per_byte,
            "margin": margin,
            "margin_per_token": margin_per_token,
            "margin_per_char": margin_per_char,
            "margin_per_byte": margin_per_byte,
            "total_prob": total_prob,
            "total_prob_per_token": total_prob_per_token,
            "total_prob_per_char": total_prob_per_char,
            "total_prob_per_byte": total_prob_per_byte,
            "uncond_correct_prob": uncond_correct_prob,
            "uncond_correct_prob_per_token": uncond_correct_prob_per_token,
            "uncond_correct_prob_per_char": uncond_correct_prob_per_char,
            "uncond_correct_prob_per_byte": uncond_correct_prob_per_byte,
            "uncond_total_prob": uncond_total_prob,
            "norm_correct_prob": norm_correct_prob,
            "norm_correct_prob_per_token": norm_correct_prob_per_token,
            "norm_correct_prob_per_char": norm_correct_prob_per_char,
            "norm_correct_prob_per_byte": norm_correct_prob_per_byte,
        }
        metrics.update(row_dict)

    # 2. Generation tasks
    elif any(key in metrics for key in generate_primary_metrics):

        # Case: Codex - Check if model_outputs has 2 elements
        if len(model_outputs) == 2:
            model_outputs = model_outputs[:1]  # pass_at_1

        if len(model_outputs) > 1:
            raise ValueError(
                "Assume generation tasks only have one output (greedy): ",
                len(model_outputs),
            )

        logits = model_outputs[0]["sum_logits"]
        num_tokens = (
            model_outputs[0]["num_tokens"] if model_outputs[0]["num_tokens"] > 0 else 1
        )
        num_chars = (
            len(model_outputs[0]["continuation"])
            if model_outputs[0]["continuation"]
            else 1
        )

        logit_per_token = logits / num_tokens
        logit_per_char = logits / num_chars

        # Case: sum_scores only available in latest version
        if "sum_scores" in model_outputs[0]:
            scores = model_outputs[0]["sum_scores"]
            score_per_token = scores / num_tokens
            score_per_char = scores / num_chars
            check_finite_and_nan(logits, "logit")
            check_finite_and_nan(logit_per_token, "logit_per_token")
            check_finite_and_nan(logit_per_char, "logit_per_char")
        else:
            scores = None
            score_per_token = None
            score_per_char = None

        row_dict = {
            "logit": logits,
            "logit_per_token": logit_per_token,
            "logit_per_char": logit_per_char,
            "score": scores,
            "score_per_token": score_per_token,
            "score_per_char": score_per_char,
        }
        metrics.update(row_dict)

    return metrics


def compute_metrics_from_file(predictions_content: str, task=None) -> list:
    """
    Read predictions from a JSONL file and compute various metrics for each task sample.

    Args:
        predictions_content (str): Content of the predictions file in JSONL format.

    Returns:
        list: A list of dictionaries containing computed metrics for each task sample.
    """
    predictions = [json.loads(line) for line in predictions_content.splitlines()]
    rows = []

    for prediction in predictions:
        metrics = process_predictions_cheap_decisions(prediction, task=task)
        rows.append(metrics)

    return rows


def parse_train_name(path):
    """
    Parse the S3 path to extract the group, model, chinchilla, task, and step.
    Example input path structure: "checkpoints/benb/olmo-150M-no_math_no_code-1xC/step500/mmlu/predictions.jsonl"
    """
    parts = path.split("/")[7:]
    assert re.match(
        r".*-\d+xC(-\d+)?$", parts[3]
    ), f"Invalid model name format: {parts[3]}"

    if re.match(r".*-\d+xC-\d+$", parts[3]):
        group_model_chinchilla = parts[3].rsplit("-", 3)
        seed = group_model_chinchilla[3]
        assert re.match(
            r"\d", seed
        ), f"Invalid model name parsing: {parts[3]} -> {group_model_chinchilla}"
        seed = int(seed)
    elif re.match(r".*-\d+xC$", parts[3]):
        group_model_chinchilla = parts[3].rsplit("-", 2)
        seed = None
    else:
        raise ValueError(f"Invalid model name format: {parts}")

    group = group_model_chinchilla[0]
    model = group_model_chinchilla[1]
    assert re.match(r"\d+[M|B]", model), f"Invalid model size parsing: {model}"
    chinchilla = group_model_chinchilla[2]
    assert re.match(r"\d+xC", chinchilla), f"Invalid chinchilla parsing: {chinchilla}"
    step = int(re.search(r"step(\d+)", parts[4]).group(1))
    if "all_olmes" in path:
        task = None
        if "_rc_tasks" in path:
            metrics_match = re.search(r"task-\d+-(.*?)-metrics\.json", parts[6])
            predictions_match = re.search(r"task-\d+-(.*?)-predictions\.jsonl", parts[6])
            if metrics_match:
                task = metrics_match.group(1)
            elif predictions_match:
                task = predictions_match.group(1)
    else:
        task = parts[5]
    return group, model, chinchilla, task, step, seed

import numpy as np
from typing import Dict, Any

def debug_aggregation(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Debug metrics aggregation with detailed error reporting.
    
    Args:
        metrics_dict: Dictionary of metrics to aggregate
    Returns:
        Dictionary of mean values for each metric
    """
    mean_metrics = {}
    
    for key, values in metrics_dict.items():
        try:
            # Try to convert to numpy array and get mean
            arr = np.array(values)
            try:
                mean_metrics[key] = np.mean(arr)
            except Exception as e:
                print(f'Couldnt divide on {key}: {arr}')
                mean_metrics[key] = 0
            
        except ValueError as e:
            pass
            
    return mean_metrics


def process_prediction_path(path, rows_list):
    group, model, chinchilla, task, step, seed = parse_train_name(path)

    # # Extract metrics
    # # rows_list = compute_metrics_from_file(predictions, task)
    # rows_list = []
    # for prediction in predictions:
    #     metrics = compute_metrics_from_file(prediction, task=task)
    #     rows_list.append(metrics)
    # if rows_list is None:
    #     print(f"Skipping results for: {task}")
    #     return None

    if task not in PRIMARY_METRICS_OLMES:
        raise RuntimeError(f'Could not find "{task}" on path {path}!')
    primary_metric = PRIMARY_METRICS_OLMES[task]
    
    # Get primary_metric in this order
    possible_metrics = [
        "primary_metric",
        "acc_raw",
        "exact_match",
        "f1",
        "mc1",
        "pass_at_1",
        "prompt_level_loose_acc",
        "maj_at_1",
    ]

    aggregated_metrics = {}
    for mrow in rows_list:
        if "em" in mrow:
            mrow["exact_match"] = mrow.pop("em")
        if primary_metric is None:
            for metric in possible_metrics:
                if metric in mrow:
                    # Set name forprimary_metric
                    primary_metric = metric
                    break
        if primary_metric is None:
            print(f"Skipping task {task} due to missing primary metric: {mrow}")
            continue

        mrow["primary_metric"] = mrow[primary_metric]
        mrow["acc_raw"] = mrow["acc_raw"]
        mrow["acc_per_char"] = mrow["acc_per_char"]
        mrow["acc_per_token"] = mrow["acc_per_token"]
        mrow["acc_uncond"] = mrow["acc_uncond"]
        for key, value in mrow.items():
            if value is None or isinstance(value, str):
                continue
            if key in aggregated_metrics:
                aggregated_metrics[key].append(value)
            else:
                aggregated_metrics[key] = [value]

    # mean_metrics = {k: np.mean(v) for k, v in aggregated_metrics.items()}
    mean_metrics = debug_aggregation(aggregated_metrics)

    row = {
        "group": group,
        "model": model,
        "task": task,
        "chinchilla": chinchilla,
        "step": step,
        "seed": seed,
        "metrics": mean_metrics,
    }
    return row


def define_compute_proportions(steps=5):
    return [i / steps for i in range(1, steps + 1)]


def unpack_dict_column(df, col_name):
    """
    Unpack a dictionary column in a DataFrame using json_normalize.
    Return a new DataFrame with the unpacked columns joined.
    """
    temp = pd.json_normalize(df[col_name], max_level=1)
    temp = temp.reset_index(drop=True)
    df = df.reset_index(drop=True).drop(columns=[col_name]).join(temp)
    # print(f"Columns from unpacking: {df.columns}")
    return df


def format_tokens(tokens: int):
    if tokens >= 1_000_000_000:  # Check for billions
        return f"{tokens / 1_000_000_000:.1f}B"
    elif tokens >= 1_000_000:  # Check for millions
        return f"{tokens / 1_000_000:.1f}M"
    else:
        return str(tokens)


def find_common_checkpoints(metric_values1: np.array, metric_values2: np.array):
    """Find all common checkpoints between two metric arrays."""
    # Identify non-NaN indices for both arrays
    valid_indices1 = ~np.isnan(metric_values1)
    valid_indices2 = ~np.isnan(metric_values2)
    common_indices = np.where(valid_indices1 & valid_indices2)[0]
    if not len(common_indices):
        raise ValueError("No common checkpoints found between the two mixes.")
    return common_indices


def clean_nans(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    mask = np.isfinite(arr1) & np.isfinite(arr2)
    # Check if any NaNs were removed by comparing the original and filtered lengths
    changed = not np.all(mask)
    # Apply the mask to filter out NaN indices
    filtered_arr1 = arr1[mask].tolist()
    filtered_arr2 = arr2[mask].tolist()
    return filtered_arr1, filtered_arr2, changed

import ast

def safe_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error processing value: {x} | Error: {e}")
        return x  # Keep the original value if it fails


def expand_df(df, quiet=True):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')

    # Preprocess the df into a usuable format
    if not quiet: print('Converting metrics dict to a set of cols...')
    # df["metrics"] = df["metrics"].apply(eval)
    df["metrics"] = df["metrics"].apply(safe_eval)
    metrics_df = df["metrics"].apply(pd.Series)
    df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove bad mixes
    BAD_MIXES = ["DCLM-baseline-25p", "DCLM-baseline-50p", "DCLM-baseline-75p"]
    for bad_mix in BAD_MIXES:
        df = df[df["group"] != bad_mix]

    df.loc[df['group'] == 'baseline', 'group'] = 'dolma17'

    df['size'] = df['model']
    df['model'] = df['group'] + '-' + df['model'] + '-' + df['chinchilla']

    return df


def clean_data_and_compute_averages(df, quiet=True):
    """ Wrapper around Ian's data cleaning to compute macro averages """
    if not quiet: print('Launching data cleaning!')

    df = ian_clean_data(df, dirty_out=False, quiet=quiet)

    if not quiet: print('Computing macro averages...')

    print(f'Before computing macro averages: {len(df)}')

    # # Compute MMLU macro-average
    # group_cols = ['group', 'model', 'chinchilla', 'step', 'seed']
    # agg_cols = [col for col in df.columns if col not in group_cols and col != 'task']
    # mmlu_rows = df[df['task'].str.contains("MMLU", case=False)]
    # numeric_cols = mmlu_rows[agg_cols].select_dtypes(include=['number']).columns.tolist()
    # aggregated = mmlu_rows.groupby(group_cols, as_index=False)[numeric_cols].mean()
    # aggregated['task'] = 'mmlu'
    # df = df[~df['task'].str.contains("MMLU", case=False)]
    # df = pd.concat([df, aggregated], ignore_index=True)

    # # Compute olmes macro-average
    # group_cols = ['group', 'model', 'chinchilla', 'step', 'seed']
    # agg_cols = [col for col in df.columns if col not in group_cols and col != 'task']
    # olmes_rows = df # olmes_rows = df[df['task'].str.contains("olmes", case=False)]
    # numeric_cols = olmes_rows[agg_cols].select_dtypes(include=['number']).columns.tolist()
    # aggregated = olmes_rows.groupby(group_cols, as_index=False)[numeric_cols].mean()
    # aggregated['task'] = 'olmes_10_macro_avg'
    # df = pd.concat([df, aggregated], ignore_index=True)

    group_cols = ['group', 'model', 'chinchilla', 'step', 'seed']
    agg_cols = [col for col in df.columns if col not in group_cols and col not in ['task', 'num_instances']]

    # Compute MMLU macro-average
    mmlu_rows = df[df['task'].str.contains("MMLU", case=False)]
    numeric_cols = mmlu_rows[agg_cols].select_dtypes(include=['number']).columns.tolist()
    aggregated = mmlu_rows.groupby(group_cols, as_index=False).agg({col: 'mean' for col in numeric_cols})
    aggregated['num_instances'] = mmlu_rows.groupby(group_cols)['num_instances'].sum().values
    aggregated['task'] = 'mmlu'
    df = df[~df['task'].str.contains("MMLU", case=False)]
    df = pd.concat([df, aggregated], ignore_index=True)

    # Compute Olmes macro-average
    olmes_rows = df  # If needed, filter with df[df['task'].str.contains("olmes", case=False)]
    numeric_cols = olmes_rows[agg_cols].select_dtypes(include=['number']).columns.tolist()
    aggregated = olmes_rows.groupby(group_cols, as_index=False).agg({col: 'mean' for col in numeric_cols})
    aggregated['num_instances'] = olmes_rows.groupby(group_cols)['num_instances'].sum().values
    aggregated['task'] = 'olmes_10_macro_avg'
    df = pd.concat([df, aggregated], ignore_index=True)

    print(f'After computing macro averages: {len(df)}')

    df['size'] = df['model'].str.split('-').str[-2]

    # Remove extra metrics columns that were not used everywhere
    df = df.drop(columns=[
        "predicted_index_per_byte", 
        "acc_per_byte", 
        "sum_logits_corr", 
        "logits_per_token_corr", 
        "logits_per_char_corr", 
        "logits_per_byte_corr"
    ], errors='ignore')

    if not quiet: print('Done!')

    return df


def ian_clean_data(df, dirty_out=False, quiet=True):
    """ Clean data according to https://github.com/allenai/oe-eval-internal/blob/eval-for-consistent-ranking/experiments/eval-for-consistent-ranking/metrics/project/data_exploration_and_cleaning.ipynb """
    
    print(f'Step 0: {len(df)}')
    
    # Ian uses only the size for "model"
    if 'model_full' not in df.columns:
        df['model_full'] = df['model']
        df['model'] = df['model'].apply(lambda x: x.split('-')[-2] if '-' in x else None)
    
    # 1) Clean group names
    # print(len(df['group'].unique()))
    df.loc[df['group'] == 'baseline', 'group'] = 'dolma17'

    bad_mixes = [
        'DCLM-baseline-25p',
        'DCLM-baseline-50p',
        'DCLM-baseline-75p',
    ]

    cannonical_groups = set([
        'DCLM-baseline',
        'c4',
        'dclm_ft7percentile_fw2',
        'dclm_ft7percentile_fw3',
        'dclm_fw_top10',
        'dclm_fw_top3',
        'dolma-v1-6-and-sources-baseline',
        'dolma17',
        'dolma17-25p-DCLM-baseline-75p',
        'dolma17-50p-DCLM-baseline-50p',
        'dolma17-75p-DCLM-baseline-25p',
        'falcon',
        'falcon_and_cc',
        'falcon_and_cc_eli5_oh_top10p',
        'falcon_and_cc_eli5_oh_top20p',
        'falcon_and_cc_og_eli5_oh_top10p',
        'falcon_and_cc_tulu_qc_top10',
        'fineweb_edu_dedup',
        'no_code',
        'no_flan',
        'no_math_no_code',
        'no_reddit',
        'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p',
        'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p',
        'prox_fineweb_pro'
    ])

    df = df[~df['group'].isin(bad_mixes)]
    df.loc[df['group'] == 'DCLM-baseline-4M-5xC', 'group'] = 'DCLM-baseline'
    df = df[df['model_full'] != 'DCLM-baseline-4M-5xC-4M-5xC']

    print(df['group'].unique())
    print(df['group'].value_counts())
    print(df['size'].value_counts())

    # print(len(df['group'].unique()))
    assert set(sorted(df['group'].unique())) == cannonical_groups

    print(f'Step 1: {len(df)}')

    # 2) Normalize seeds
    if not quiet:
        print('raw seeds:')
        print(df['seed'].unique())

    def normalize_seeds(df):
        df = df.copy()
        df['seed'] = df['seed'].fillna(6198)
        df['seed'] = df['seed'].astype(int)
        df.loc[df['seed'] == 2, 'seed'] = 6198
        return df

    df = normalize_seeds(df)

    if not quiet:
        print('normalized seeds:')
        print(df['seed'].unique())

    print(f'Step 2: {len(df)}')

    # 3) Throw out 1B seed 14 and 15 cuz we have full seed runs from 4 and 5
    df = df[~((df['model'] == '1B') & (df['seed'].isin([14, 15])))]

    print(f'Step 3: {len(df)}')

    # # 4) Remove all steps without 3 seeds
    # if not dirty_out:
    #     pre_filter_groups = df['group'].unique()
    #     filtered_groups = []
    #     throwaway_groups = []
    #     for name, data in df.groupby(['model', 'group', 'step']): # understand why token and compute effecting this so much? ['model', 'group', 'step', 'tokens', 'compute']
    #         if len(set(data['seed'])) == 3:
    #             filtered_groups.append(data)
    #         else:
    #             throwaway_groups.append(data)
    #     df = pd.concat(filtered_groups)
    #     df_throwaway = pd.concat(throwaway_groups)
    #     df_throwaway['group'].value_counts()

    #     if not quiet: print(len(df['group'].unique()))
    #     post_filter_groups = df['group'].unique()
    #     if not quiet: print(set(pre_filter_groups) - set(post_filter_groups))

    #     # are any steps missing some groups?
    #     missing_groups = []
    #     for (model, step), data in df.groupby(['model', 'step']):
    #         present_groups = set(data['group'].unique())
    #         missing = set(post_filter_groups) - present_groups
    #         if missing:
    #             missing_groups.append((model, step, missing))

    print(f'Step 4: {len(df)}')
    
    # 5) Throw out all steps beyond the end of LR schedule
    full_schedule_last_step_per_model = {
        '4M': 5725,  # min step value from {5745, 5725, 5735}
        '20M': 14584,  # min step value from {14584, 14594}
        '60M': 29042,  # min step value from {29042, 29052, 29062}
        '90M': 29901,
        '150M': 38157,
        '300M': 45787,
        '530M': 57786,
        '750M': 63589,
        '1B': 69369,
    }

    # have to round up for the models that ran to long to make sure we get a checkpoint after the LR fully decays
    def round_up(value, increment):
        return (value + increment - 1) // increment * increment
    for model, step in full_schedule_last_step_per_model.items():
        if model == '1B':
            full_schedule_last_step_per_model[model] = round_up(step, 2500)
        else:
            full_schedule_last_step_per_model[model] = round_up(step, 1250)

    df = df[df.apply(lambda row: row['step'] <= full_schedule_last_step_per_model[row['model']], axis=1)]

    # are there some groups that have fewer steps for a given model size?
    min_last_step_per_model = {}
    for name, data in df.groupby(['model', 'seed']):
        max_per_model = -1
        max_steps = None
        max_group = None
        min_per_model = 100000
        min_group = None
        min_steps = None
        for group in data['group'].unique():
            group_data = data[data['group'] == group]
            if len(group_data['step'].unique()) > max_per_model:
                max_per_model = len(group_data['step'].unique())
                max_group = group
                max_steps = sorted(group_data['step'].unique())
            if len(group_data['step'].unique()) < min_per_model:
                min_per_model = len(group_data['step'].unique())
                min_group = group
                min_steps = sorted(group_data['step'].unique())
        min_last_step_per_model[name] = min_steps[-1]
        if not quiet:
            if max_per_model != min_per_model:
                print(f"max steps for {name}: {max_per_model}")
                print(f"min steps for {name}: {min_per_model}")
                print(f"min group for {name}: {min_group}")


    print(f'Step 5: {len(df)}')
    
    # 6) [RESOLVED] remove groups that don't have targets for 3 seeds for final result
    target_df = df[df['model'] == '1B']
    group_seeds = {}
    for _, row in target_df[['group', 'seed']].iterrows():
        group = row['group']
        seed = row['seed']
        if group not in group_seeds:
            group_seeds[group] = set()
        group_seeds[group].add(seed)

    # group_seeds

    for group in target_df['group'].unique():
        # assert len(target_df['seed'].unique()) == 1, f"Uncomment the next line for more seeds: {target_df['seed'].value_counts()}"
        for seed in {4, 5, 6198}:
        # for seed in {4}:
            latest_step = target_df[(target_df['group'] == group) & (target_df['seed'] == seed)]['step'].max()
            assert latest_step == 69369, f"seed {seed} latest step: {latest_step}"
            if latest_step != 69369:
                print(f"seed {seed} latest step: {latest_step}")

    print(f'Step 6 (excluded): {len(df)}')

    # df = df[df["sum_logits_corr"] != 0]
    df = df.fillna(0)
    # df = df[~df["sum_logits_corr"].between(-1e-2, 1e-2)]

    # Grouping criteria
    group_cols = ['model', 'group', 'task', 'step', 'seed']

    # Count occurrences of each group
    df['count'] = df.groupby(group_cols)['sum_logits_corr'].transform('count')

    # Identify groups with exactly two occurrences where one has sum_logits_corr == 0
    to_drop = df[(df['count'] == 2) & (df['sum_logits_corr'] == 0)][group_cols]

    # Drop all rows belonging to these groups
    df = df.merge(to_drop, on=group_cols, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['count', '_merge'])

    print(df)
    
    # 7) Throw out duplicate rows based on ['model', 'group', 'task', 'step', 'seed']
    def check_duplicates(df):
        for name, data in df.groupby(['model', 'group', 'task', 'step','seed']):
            if len(data) != 1:
                print(f"there are duplicates here in {name}: {data}")
                diff_columns = data.loc[:, (data != data.iloc[0]).any()].columns
                print(f"Different columns: {diff_columns}")
                print(f"The different values are:\n{data[diff_columns]}")
                raise

    # prove to myself that these are just duplicates before dropping them (uncomment to run)
    check_duplicates(df.round(6).drop_duplicates())

    # drop duplicates
    df = df.groupby(['model', 'group', 'task', 'step', 'seed']).first().reset_index()
    # if not dirty_out:
    #     assert all(len(d) == 1 for n,d in df.groupby(['model', 'group', 'task', 'step','seed']) ), f"There are duplicates in the data; max size per models X group X steps X seed X task was {max((len(d) for n, d in df.groupby(['model', 'group', 'task', 'step','seed'])))}"
    #     assert all(len(d) == 3 for n, d in df.groupby(['model', 'group', 'task', 'step'])['seed']), f"Not all models X group X steps X task have 3 seeds; min size was {df.groupby(['model', 'group', 'step'])['seed'].size().min()}"
    #     assert all(d['seed'].nunique() ==3 for n, d in df.groupby(['model', 'group', 'task', 'step'])), f"Not all models X group X steps X task have 3 seeds; min size was {df.groupby(['model', 'group', 'step'])['seed'].nunique().min()}"

    print(f'Step 7: {len(df)}')
    
    # 8) Resolve NaNs by recomputing token and compute values for all rows
    # These columns are not really used now
    columns_to_drop = [
        'eval/c4_en-validation/CrossEntropyLoss',
        'eval/dolma_common-crawl-validation/CrossEntropyLoss',
        'eval/pile-validation/CrossEntropyLoss',
        'eval/wikitext_103-validation/CrossEntropyLoss',
        'train/CrossEntropyLoss',
        'throughput/total_tokens'
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    model_to_batch = {
        '4M': 32,
        '20M': 64,
        '60M': 96,
        '90M': 160,
        '150M': 192,
        '300M': 320,
        '530M': 448,
        '750M': 576,
        '1B': 704
    }
    
    model_to_params = {
        '4M': 3744832,
        '20M': 19101888,
        '60M': 57078144,
        '90M': 97946640,
        '150M': 151898880, 
        '300M': 319980544, 
        '530M': 530074944, 
        '750M': 681297408, 
        '1B': 1176832000, # Non embedding params
    }
    model_to_params = {k: float(v) for k, v in model_to_params.items()}

    sequence_length = 2048

    def model_and_step_to_tokens(model, step):
        return model_to_batch[model] * step * sequence_length

    def model_and_step_to_compute(model, step):
        return model_to_params[model] * model_and_step_to_tokens(model, step) * 6

    # Prove to myself that we can just recompute these values
    # for rows in df.dropna().itertuples():
    #     estimated_compute = model_and_step_to_compute(rows.model, rows.step)
    #     estimated_tokens = model_and_step_to_tokens(rows.model, rows.step)
    #     assert abs(estimated_compute - rows.compute) < 1e-6, f"Compute mismatch for model {rows.model}, step {rows.step}"
    #     assert abs(estimated_tokens - rows.tokens) < 1e-6, f"Tokens mismatch for model {rows.model}, step {rows.step}"

    def recompute_tokens_and_compute(df):
        df = df.copy()
        df['tokens'] = df.apply(lambda row: model_and_step_to_tokens(row['model'], row['step']), axis=1)
        df['compute'] = df.apply(lambda row: model_and_step_to_compute(row['model'], row['step']), axis=1)
        return df

    df = recompute_tokens_and_compute(df)

    print(f'Step 8: {len(df)}')
    
    # 9) Remove all steps that don't have all groups
    if dirty_out:
        df = df[df['seed'] == 6198]

    def remove_incomplete_model_steps(df):
        available_groups = set(df.group.unique())
        df = df.copy()
        # Group by model and step
        grouped = df.groupby(['model', 'step'])
        
        # Filter out groups that don't have all available groups
        complete_groups = [name for name, group in grouped if set(group['group'].unique()) == available_groups]
        
        # Filter the dataframe to keep only the complete groups
        filtered_df = df[df.set_index(['model', 'step']).index.isin(complete_groups)]
        
        return filtered_df

    available_groups = set(df.group.unique())
    # assert all(set(d.group.unique()) == available_groups for n, d in remove_incomplete_model_steps(df).groupby(['model', 'task', 'step', 'seed']))

    print(f'Step 9 (now included!): {len(df)}')
    
    # restore model col
    df['model'] = df['model_full']
    df.drop(columns=['model_full'], errors='ignore', inplace=True)

    return df
