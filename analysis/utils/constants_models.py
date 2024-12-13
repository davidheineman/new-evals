WEKA_CLUSTERS = ",".join(
    ["ai2/jupiter-cirrascale-2", "ai2/saturn-cirrascale"]
)
# "ai2/neptune-cirrascale", # L40s, can't load 70B+

RC_TASKS_OLMES = [
    "arc_challenge:rc::olmes:full",
    "arc_easy:rc::olmes:full",
    "boolq:rc::olmes:full",
    "csqa:rc::olmes:full",
    "hellaswag:rc::olmes:full",
    "openbookqa:rc::olmes:full",
    "piqa:rc::olmes:full",
    "socialiqa:rc::olmes:full",
    "winogrande:rc::olmes:full",
    "mmlu_abstract_algebra:rc::olmes:full",
    "mmlu_anatomy:rc::olmes:full",
    "mmlu_astronomy:rc::olmes:full",
    "mmlu_business_ethics:rc::olmes:full",
    "mmlu_clinical_knowledge:rc::olmes:full",
    "mmlu_college_biology:rc::olmes:full",
    "mmlu_college_chemistry:rc::olmes:full",
    "mmlu_college_computer_science:rc::olmes:full",
    "mmlu_college_mathematics:rc::olmes:full",
    "mmlu_college_medicine:rc::olmes:full",
    "mmlu_college_physics:rc::olmes:full",
    "mmlu_computer_security:rc::olmes:full",
    "mmlu_conceptual_physics:rc::olmes:full",
    "mmlu_econometrics:rc::olmes:full",
    "mmlu_electrical_engineering:rc::olmes:full",
    "mmlu_elementary_mathematics:rc::olmes:full",
    "mmlu_formal_logic:rc::olmes:full",
    "mmlu_global_facts:rc::olmes:full",
    "mmlu_high_school_biology:rc::olmes:full",
    "mmlu_high_school_chemistry:rc::olmes:full",
    "mmlu_high_school_computer_science:rc::olmes:full",
    "mmlu_high_school_european_history:rc::olmes:full",
    "mmlu_high_school_geography:rc::olmes:full",
    "mmlu_high_school_government_and_politics:rc::olmes:full",
    "mmlu_high_school_macroeconomics:rc::olmes:full",
    "mmlu_high_school_mathematics:rc::olmes:full",
    "mmlu_high_school_microeconomics:rc::olmes:full",
    "mmlu_high_school_physics:rc::olmes:full",
    "mmlu_high_school_psychology:rc::olmes:full",
    "mmlu_high_school_statistics:rc::olmes:full",
    "mmlu_high_school_us_history:rc::olmes:full",
    "mmlu_high_school_world_history:rc::olmes:full",
    "mmlu_human_aging:rc::olmes:full",
    "mmlu_human_sexuality:rc::olmes:full",
    "mmlu_international_law:rc::olmes:full",
    "mmlu_jurisprudence:rc::olmes:full",
    "mmlu_logical_fallacies:rc::olmes:full",
    "mmlu_machine_learning:rc::olmes:full",
    "mmlu_management:rc::olmes:full",
    "mmlu_marketing:rc::olmes:full",
    "mmlu_medical_genetics:rc::olmes:full",
    "mmlu_miscellaneous:rc::olmes:full",
    "mmlu_moral_disputes:rc::olmes:full",
    "mmlu_moral_scenarios:rc::olmes:full",
    "mmlu_nutrition:rc::olmes:full",
    "mmlu_philosophy:rc::olmes:full",
    "mmlu_prehistory:rc::olmes:full",
    "mmlu_professional_accounting:rc::olmes:full",
    "mmlu_professional_law:rc::olmes:full",
    "mmlu_professional_medicine:rc::olmes:full",
    "mmlu_professional_psychology:rc::olmes:full",
    "mmlu_public_relations:rc::olmes:full",
    "mmlu_security_studies:rc::olmes:full",
    "mmlu_sociology:rc::olmes:full",
    "mmlu_us_foreign_policy:rc::olmes:full",
    "mmlu_virology:rc::olmes:full",
    "mmlu_world_religions:rc::olmes:full",
]

MC_TASKS_OLMES = [task.replace(":rc", ":mc") for task in RC_TASKS_OLMES]
PARA_TASKS_OLMES = [task.replace(":rc", ":para") for task in RC_TASKS_OLMES]

MC_TASKS_COPY_COLORS = [
    "copycolors_2way:mc::none",
    "copycolors_cyclic_2way:mc::none",
    "copycolors_4way:mc::none",
    "copycolors_cyclic_4way:mc::none",
    "copycolors_8way:mc::none",
    "copycolors_cyclic_8way:mc::none",
]

GEN_TASKS_OLMES = [
    # Core generation-based benchmarks
    "coqa::olmes:full",
    "drop::olmes:full",
    "gsm8k::olmes:full",
    "jeopardy::olmes:full",
    "naturalqs::olmes:full",
    "squad::olmes:full",
    "triviaqa::olmes:full",
]

GEN_TASKS_EXTENDED = [
    # MC benchmarks
    "agi_eval_lsat-ar::olmes:full",
    "agi_eval_lsat-lr::olmes:full",
    "agi_eval_lsat-rc::olmes:full",
    "agi_eval_logiqa-en::olmes:full",
    "agi_eval_sat-math::olmes:full",
    "agi_eval_sat-en::olmes:full",
    "agi_eval_aqua-rat::olmes:full",
    "agi_eval_sat-en-without-passage::olmes:full",
    "agi_eval_gaokao-english::olmes:full",
    "minerva_math_algebra::olmes:full",
    "minerva_math_counting_and_probability::olmes:full",
    "minerva_math_geometry::olmes:full",
    "minerva_math_intermediate_algebra::olmes:full",
    "minerva_math_number_theory::olmes:full",
    "minerva_math_prealgebra::olmes:full",
    "minerva_math_precalculus::olmes:full",

    # COT benchmarks (generation-based) -- also :qa::none varaints
    "bbh_boolean_expressions:cot::olmes:full",
    "bbh_causal_judgement:cot::olmes:full",
    "bbh_date_understanding:cot::olmes:full",
    "bbh_disambiguation_qa:cot::olmes:full",
    "bbh_dyck_languages:cot::olmes:full",
    "bbh_formal_fallacies:cot::olmes:full",
    "bbh_geometric_shapes:cot::olmes:full",
    "bbh_hyperbaton:cot::olmes:full",
    "bbh_logical_deduction_five_objects:cot::olmes:full",
    "bbh_logical_deduction_seven_objects:cot::olmes:full",
    "bbh_logical_deduction_three_objects:cot::olmes:full",
    "bbh_movie_recommendation:cot::olmes:full",
    "bbh_multistep_arithmetic_two:cot::olmes:full",
    "bbh_navigate:cot::olmes:full",
    "bbh_object_counting:cot::olmes:full",
    "bbh_penguins_in_a_table:cot::olmes:full",
    "bbh_reasoning_about_colored_objects:cot::olmes:full",
    "bbh_ruin_names:cot::olmes:full",
    "bbh_salient_translation_error_detection:cot::olmes:full",
    "bbh_snarks:cot::olmes:full",
    "bbh_sports_understanding:cot::olmes:full",
    "bbh_temporal_sequences:cot::olmes:full",
    "bbh_tracking_shuffled_objects_five_objects:cot::olmes:full",
    "bbh_tracking_shuffled_objects_seven_objects:cot::olmes:full",
    "bbh_tracking_shuffled_objects_three_objects:cot::olmes:full",
    "bbh_web_of_lies:cot::olmes:full",
    "bbh_word_sorting:cot::olmes:full",
]

BBH_QA = [
    # QA Variants of BBH Tasks
    "bbh_boolean_expressions:qa::none",
    "bbh_causal_judgement:qa::none",
    "bbh_date_understanding:qa::none",
    "bbh_disambiguation_qa:qa::none",
    "bbh_dyck_languages:qa::none",
    "bbh_formal_fallacies:qa::none",
    "bbh_geometric_shapes:qa::none",
    "bbh_hyperbaton:qa::none",
    "bbh_logical_deduction_five_objects:qa::none",
    "bbh_logical_deduction_seven_objects:qa::none",
    "bbh_logical_deduction_three_objects:qa::none",
    "bbh_movie_recommendation:qa::none",
    "bbh_multistep_arithmetic_two:qa::none",
    "bbh_navigate:qa::none",
    "bbh_object_counting:qa::none",
    "bbh_penguins_in_a_table:qa::none",
    "bbh_reasoning_about_colored_objects:qa::none",
    "bbh_ruin_names:qa::none",
    "bbh_salient_translation_error_detection:qa::none",
    "bbh_snarks:qa::none",
    "bbh_sports_understanding:qa::none",
    "bbh_temporal_sequences:qa::none",
    "bbh_tracking_shuffled_objects_five_objects:qa::none",
    "bbh_tracking_shuffled_objects_seven_objects:qa::none",
    "bbh_tracking_shuffled_objects_three_objects:qa::none",
    "bbh_web_of_lies:qa::none",
    "bbh_word_sorting:qa::none",
]

PERTURB_COT_TASKS = [
    'agi_eval_aqua-rat:perturb_cot::olmes', 
    'agi_eval_gaokao-english:perturb_cot::olmes', 
    'agi_eval_logiqa-en:perturb_cot::olmes', 
    'agi_eval_lsat-ar:perturb_cot::olmes', 
    'agi_eval_lsat-lr:perturb_cot::olmes', 
    'agi_eval_lsat-rc:perturb_cot::olmes', 
    'agi_eval_sat-en-without-passage:perturb_cot::olmes', 
    'agi_eval_sat-en:perturb_cot::olmes', 
    'agi_eval_sat-math:perturb_cot::olmes', 
    'bbh_boolean_expressions:perturb_cot::olmes', 
    'bbh_causal_judgement:perturb_cot::olmes', 
    'bbh_date_understanding:perturb_cot::olmes', 
    'bbh_disambiguation_qa:perturb_cot::olmes', 
    'bbh_dyck_languages:perturb_cot::olmes', 
    'bbh_formal_fallacies:perturb_cot::olmes', 
    'bbh_geometric_shapes:perturb_cot::olmes', 
    'bbh_hyperbaton:perturb_cot::olmes', 
    'bbh_logical_deduction_five_objects:perturb_cot::olmes', 
    'bbh_logical_deduction_seven_objects:perturb_cot::olmes', 
    'bbh_logical_deduction_three_objects:perturb_cot::olmes', 
    'bbh_movie_recommendation:perturb_cot::olmes', 
    'bbh_multistep_arithmetic_two:perturb_cot::olmes', 
    'bbh_navigate:perturb_cot::olmes', 
    'bbh_object_counting:perturb_cot::olmes', 
    'bbh_penguins_in_a_table:perturb_cot::olmes', 
    'bbh_reasoning_about_colored_objects:perturb_cot::olmes', 
    'bbh_ruin_names:perturb_cot::olmes', 
    'bbh_salient_translation_error_detection:perturb_cot::olmes', 
    'bbh_snarks:perturb_cot::olmes', 
    'bbh_sports_understanding:perturb_cot::olmes', 
    'bbh_temporal_sequences:perturb_cot::olmes', 
    'bbh_tracking_shuffled_objects_five_objects:perturb_cot::olmes', 
    'bbh_tracking_shuffled_objects_seven_objects:perturb_cot::olmes', 
    'bbh_tracking_shuffled_objects_three_objects:perturb_cot::olmes', 
    'bbh_web_of_lies:perturb_cot::olmes', 
    'bbh_word_sorting:perturb_cot::olmes', 
    'gsm8k:perturb_cot::olmes', 
    'minerva_math_algebra:perturb_cot::olmes', 
    'minerva_math_counting_and_probability:perturb_cot::olmes', 
    'minerva_math_geometry:perturb_cot::olmes', 
    'minerva_math_intermediate_algebra:perturb_cot::olmes', 
    'minerva_math_number_theory:perturb_cot::olmes', 
    'minerva_math_prealgebra:perturb_cot::olmes', 
]

COT_TASKS_TULU_3 = [
    # (Don't worry about these for now, probably not well suited for pre-training)

    # CoT (MC) -- same as OLMES Minevera
    # "hendrycks_math_algebra::original",
    # "hendrycks_math_counting_and_probability::original",
    # "hendrycks_math_geometry::original",
    # "hendrycks_math_intermediate_algebra::original",
    # "hendrycks_math_number_theory::original",
    # "hendrycks_math_prealgebra::original",
    # "hendrycks_math_precalculus::original",

    # CoT (exact match)
    "tydiqa_english::tulu",
    "ifeval::tulu",
    "truthfulqa::tulu",
    "alpaca_eval_v2::tulu",

    # CoT Long Context (exact match)
    # "zero_scrolls_gov_report::tulu",
    # "zero_scrolls_summ_screen_fd::tulu",
    # "zero_scrolls_qmsum::tulu",
    # "zero_scrolls_qasper::tulu",
    # "zero_scrolls_narrative_qa::tulu",
    # "zero_scrolls_quality::tulu",

    # Code benchmarks
    "bigcodebench::tulu",
    "bigcodebench_hard::tulu",
    "codex_humaneval:temp0.1",
    "codex_humaneval:temp0.8",
    "codex_humaneval::tulu",
    "codex_humanevalplus::tulu"
]

# Varying the model size
MODEL_LADDER_LIST = [
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-0.5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-0.5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-0.5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-0.5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-0.5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-370M-1xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-760M-1xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-1B-1xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-3B-1xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-2xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-2xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-2xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-2xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-5xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-10xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-10xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-10xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-10xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-10xC",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646-hf-vllm",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm",
]

# Varying the checkpoint at 1B 5xC (data mix is olmoe)
MODEL_LIST_INTERMEDIATE = [
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step0-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step1000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step1500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step2000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step2500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step3000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step3500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step4000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step4500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step5000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step5500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step6000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step6500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step7000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step7500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step8000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step8500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step9000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step9500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step10000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step10500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step11000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step11500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step12000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step12500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step13000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step13500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step14000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step14500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step15000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step15500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step16000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step16500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step17000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step17500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step18000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step18500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step19000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step19500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step20000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step20500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step21000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step21500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step22000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step22500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step23000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step23500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step24000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step24500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step25000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step25500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step26000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step26500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step27000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step27500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step28000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step28500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step29000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step29500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step30000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step30500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step31000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step31500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step32000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step32500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step33000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step33500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step34000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step34500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step35000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step35500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step36000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step36500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step37000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step37500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step38000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step38500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step39000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step39500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step40000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step40500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step41000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step41500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step42000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step42500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step43000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step43500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step44000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step44500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step45000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step45500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step46000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step46500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step47000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step47500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step48000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step48500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step49000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step49500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step50000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step50500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step51000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step51500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step52000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step52500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step53000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step53500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step54000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step54500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step55000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step55500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step56000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step56500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step57000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step57500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step58000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step58500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step59000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step59500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step60000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step60500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step61000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step61500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step62000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step62500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step63000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step63500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step64000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step64500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step65000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step65500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step66000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step66500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step67000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step67500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step68000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step68500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step69000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step69500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step70000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step70500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step71000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step71500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step72000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step72500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step73000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step73500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step74000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step74500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step75000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step75500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step76000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step76500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step77000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step77500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step78000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step78500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step79000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step79500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step80000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step80500-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81000-unsharded-hf",
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf",
]

# Varying the data mix at 1B 5xC
MODEL_LIST_MIXES = [
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/baseline-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/c4-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/falcon-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/falcon_and_cc-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/falcon_and_cc_eli5_oh_top10p-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/falcon_and_cc_eli5_oh_top20p-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/falcon_and_cc_og_eli5_oh_top10p-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/falcon_and_cc_tulu_qc_top10-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/fineweb_edu_dedup-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/no_code-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/no_flan-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/no_math_no_code-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/no_reddit-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/redpajama-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/ianm/DCLM-baseline-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/ianm/dolma17-25p-DCLM-baseline-75p-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/ianm/dolma17-50p-DCLM-baseline-50p-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/ianm/dolma17-75p-DCLM-baseline-25p-1B-5xC",
    "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/ianm/dolma-v1-6-and-sources-baseline-1B-5xC",
]

# Officially supported models in oe-eval as of 12/6/2024
OE_EVAL_OFFICIAL_MODELS = [
    # "amber-7b",
    # "dclm-1b", # <- would be nice to have!
    # "dclm-7b", # <- would be nice to have!
    # "dclm-7b-instruct",
    "deepseek-7b",
    # "deepseek-v2-lite-instruct",
    "falcon-7b",
    # "falcon-rw-7b",
    "gemma-2b",
    "gemma-7b",
    "gemma2-2b",
    "gemma2-9b",
    # "gemma2-2b-instruct",
    # "gemma2-9b-instruct",
    # "gemma2-9b-instruct-SimPO",
    "llama2-7b",
    "llama2-13b",
    "llama3-8b",
    "llama3.1-8b",
    "llama3.2-1b",
    "llama3.2-3b",
    # "llama3.2-1b-instruct",
    # "llama3.2-3b-instruct",
    # "llama3-8b-instruct",
    # "llama3.1-8b-instruct",
    # "llama-3.1-tulu-2-8b",
    # "llama-3.1-tulu-2-dpo-8b",
    "llama3-70b",
    "llama3.1-70b",
    # "llama3.1-70b-instruct",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    # "mistral-nemo-base-2407-12b",
    # "mistral-nemo-base-2407-12b-instruct",
    # "mixtral-8x7b-v0.1",
    # "mixtral-8x22b-v0.1",
    # "ministral-8b-instruct-2410",
    "mpt-1b-rpj-200b",
    "mpt-7b",
    # "mpt-7b-instruct",
    "neo-7b",
    "olmo-1b",
    "olmo-1b-0724",
    "olmo-7b",
    "olmo-7b-0424",
    "olmo-7b-0724",
    # "olmo-7b-0724-instruct",
    "olmoe-1b-7b-0924",
    # "olmoe-1b-7b-0924-instruct",
    # "persimmon-8b-base",
    # "persimmon-8b-chat",
    "phi-1.5",
    "pythia-160m",
    "pythia-1b",
    "pythia-6.9b",
    "qwen2-1.5b",
    "qwen2-7b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "qwen2.5-72b",
    # "qwen2.5-7b-instruct",
    # "qwen2.5-14b-instruct",
    # "rpj-incite-7b",
    # "stablelm-2-1_6b",
    # "stablelm-2-12b",
    # "tinyllama-1.1b-3T",
    # "tulu-2-dpo-7b",
    # "xgen-7b-4k-base",
    # "xgen-7b-8k-inst",
    # "zamba2-7b",
    # "zamba2-7b-instruct",
    # "zephyr-7b-beta",
    # "gpt-3.5-turbo-0125",
    # "gpt-4o-mini-2024-07-18",
    # "gpt-4o-2024-08-06",
    # "claude-3-5-sonnet-20241022",
    # "claude-3-5-haiku-20241022",
    # "gemini-1.5-flash-002",
    # "llama-7b", # <- would be nice to have!
    # "olmo-1.7-flanfix-7b",
    # "olmo-1.7-2.7T-S3-7b",
    # "olmo-1.7-2.75T-anneal-7b",
    # "olmo-7b-amberish7-anneal-from477850-50B",
    # "openelm-3b-BUGGY",
    # "olmo-7b-amberish7-step477850-hf-olmo",
    # "olmo-1b-newhp-newds-hf-olmo",
    # "olmo-1b-newhp-newds-cx5-hf-olmo",
    # "olmo-1b-newhp-newds-cx5-datafix-hf-olmo",
    # "olmo-1b-newhp-newds-cx5-flan-hf-olmo",
    # "olmo-1b-newhp-newds-cx5-reddit-hf-olmo",
    # "olmo-1b-newhp-oldds-hf-olmo",
    # "olmo-1b-newhp-oldds-cx5-hf-olmo",
    # "olmo-7b-amberish",
    # "olmo-7b-1124-preanneal",
    # "olmo-7b-1124-preanneal-vllm",
    # "olmo-7b-peteish-anneal-from-928646-50B-no-warmup",
    # "olmo-7b-peteish-anneal-from-928646-50B-nowup-dclm07-fw2",
    # "olmo-7b-peteish-anneal-from-928646-50B-nowup-dclm07-flan",
    # "olmo-7b-1124-anneal",
    # "olmo-7b-1124-anneal-vllm",
    # "olmo-13b-peteish-highlr-step239000",
    # "olmo-13b-1124-anneal",
    # "olmo-13b-1124-anneal-vllm",
    # "olmo-13b-1124-anneal-50soup",
    # "olmo-13b-1124-anneal-50soup-vllm",
    # "olmo-13b-1124-preanneal",
    # "olmo-13b-1124-preanneal-vllm",
    # "tulu-L3.1-8B-v3.9-nc",
    # "tulu-L3.1-8B-v3.9-nc-1-pif_dpo",
    # "tulu-L3.1-70B-v3.8-lr_2e-6-2_epochs",
    # "tulu-L3.1-70B-v3.8-lr_2e-6-2_epochs-pif_dpo-2e-7",
]
