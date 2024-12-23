WEKA_CLUSTERS = ",".join(
    ["ai2/jupiter-cirrascale-2", "ai2/saturn-cirrascale", "ai2/ganymede-cirrascale"]
)
# "ai2/neptune-cirrascale", # L40s, can't load 70B+

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
    "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646-hf-vllm-2", # converted to new vllm format
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

# Smaller list of 85 intermediate checkpoints (every 1K steps instead of every 500 steps)
MODEL_LIST_INTERMEDIATE_SMALL = [model for model in MODEL_LIST_INTERMEDIATE if '500-unsharded' not in model]

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
OE_EVAL_BASE_MODELS = [
    # Base Models
    "deepseek-7b",
    "falcon-7b",
    "gemma-2b",
    "gemma-7b",
    "gemma2-2b",
    "gemma2-9b",
    "llama2-7b",
    "llama2-13b",
    "llama3-8b",
    "llama3.1-8b",
    "llama3.2-1b",
    "llama3.2-3b",
    "llama3-70b",
    "llama3.1-70b",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "mpt-7b",
    "neo-7b",
    "olmo-1b",
    "olmo-1b-0724",
    "olmo-7b",
    "olmo-7b-0424",
    "olmo-7b-0724",
    "olmoe-1b-7b-0924",
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
    # "mpt-1b-rpj-200b",

    # Other models (instruct, API, broken base models)

    # "amber-7b",
    # "dclm-1b", # <- would be nice to have!
    # "dclm-7b", # <- would be nice to have!
    # "dclm-7b-instruct",
    # "deepseek-v2-lite-instruct",
    # "falcon-rw-7b",
    # "mistral-nemo-base-2407-12b",
    # "mistral-nemo-base-2407-12b-instruct",
    # "mixtral-8x7b-v0.1",
    # "mixtral-8x22b-v0.1",
    # "mpt-7b-instruct",
    # "persimmon-8b-base",
    # "persimmon-8b-chat",
    # "rpj-incite-7b",
    # "stablelm-2-1_6b",
    # "stablelm-2-12b",
    # "tinyllama-1.1b-3T",
    # "tulu-2-dpo-7b",
    # "xgen-7b-4k-base",
    # "xgen-7b-8k-inst",
    # "zephyr-7b-beta",
    # "gpt-3.5-turbo-0125",
    # "gpt-4o-mini-2024-07-18",
    # "gpt-4o-2024-08-06",
    # "claude-3-5-sonnet-20241022",
    # "claude-3-5-haiku-20241022",
    # "gemini-1.5-flash-002",
    # "llama-7b", # <- would be nice to have!
    # "openelm-3b-BUGGY",
    
    # "olmo-1.7-flanfix-7b",
    # "olmo-1.7-2.7T-S3-7b",
    # "olmo-1.7-2.75T-anneal-7b",
    # "olmo-7b-amberish7-anneal-from477850-50B",
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

OE_EVAL_INSTRUCT_MODELS = [
    "gemma2-2b-instruct",
    "gemma2-9b-instruct",
    "gemma2-9b-instruct-SimPO",
    "llama3.2-1b-instruct",
    "llama3.2-3b-instruct",
    "llama3-8b-instruct",
    "llama3.1-8b-instruct",
    "llama-3.1-tulu-2-8b",
    "llama-3.1-tulu-2-dpo-8b",
    "llama3.1-70b-instruct",
    "olmo-7b-0724-instruct",
    "olmoe-1b-7b-0924-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
    "zamba2-7b",
    "zamba2-7b-instruct",
    "ministral-8b-instruct-2410",
]

OE_EVAL_ALL_MODELS = OE_EVAL_BASE_MODELS + OE_EVAL_INSTRUCT_MODELS