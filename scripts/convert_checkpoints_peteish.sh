#!/bin/bash

MODEL_LIST_INTERMEDIATE=(
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step0-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step1000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step1500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step2000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step2500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step3000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step3500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step4000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step4500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step5000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step5500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step6000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step6500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step7000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step7500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step8000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step8500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step9000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step9500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step10000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step10500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step11000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step11500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step12000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step12500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step13000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step13500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step14000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step14500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step15000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step15500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step16000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step16500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step17000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step17500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step18000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step18500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step19000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step19500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step20000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step20500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step21000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step21500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step22000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step22500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step23000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step23500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step24000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step24500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step25000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step25500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step26000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step26500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step27000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step27500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step28000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step28500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step29000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step29500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step30000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step30500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step31000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step31500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step32000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step32500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step33000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step33500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step34000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step34500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step35000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step35500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step36000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step36500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step37000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step37500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step38000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step38500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step39000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step39500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step40000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step40500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step41000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step41500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step42000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step42500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step43000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step43500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step44000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step44500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step45000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step45500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step46000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step46500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step47000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step47500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step48000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step48500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step49000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step49500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step50000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step50500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step51000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step51500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step52000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step52500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step53000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step53500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step54000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step54500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step55000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step55500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step56000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step56500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step57000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step57500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step58000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step58500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step59000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step59500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step60000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step60500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step61000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step61500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step62000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step62500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step63000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step63500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step64000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step64500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step65000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step65500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step66000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step66500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step67000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step67500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step68000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step68500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step69000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step69500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step70000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step70500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step71000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step71500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step72000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step72500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step73000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step73500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step74000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step74500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step75000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step75500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step76000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step76500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step77000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step77500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step78000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step78500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step79000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step79500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step80000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step80500-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81000-unsharded"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded"
)

MODEL_LIST_LADDER=(
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-0.5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-0.5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-0.5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-10xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-10xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-10xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-2xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-2xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-370M-5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-760M-5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-0.5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-0.5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-10xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-10xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-1B-1xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-1xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-1xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-370M-1xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-2xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-2xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-3B-1xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-3B-5xC"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-760M-1xC"
)

MODEL_LIST_CUSTOM=(
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-10xC/step162694-unsharded"
)

# Dolma 2 Tokenizer
TOKENIZER_PATH=/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/latest/tokenizer.json

process_model_checkpoint() {
    INPUT_DIR=$1
    OUTPUT_DIR=$1-hf

    # Skip if OUTPUT_DIR already exists
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "Output directory exists: $OUTPUT_DIR. Skipping..."
        return
    fi

    echo $INPUT_DIR
    echo $OUTPUT_DIR

    python olmo-repos/OLMo/scripts/convert_olmo2_to_hf.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --tokenizer_json_path "$TOKENIZER_PATH"
}

for MODEL in "${MODEL_LIST_LADDER[@]}"; do
    # for MODEL in "${MODEL_LIST_CUSTOM[@]}"; do
    if [[ $MODEL == *"step"* ]]; then
        # Directly process the provided model path
        process_model_checkpoint "$MODEL"
    else
        # Process the last 6 model checkpoints
        FINAL_CKPTS=$(ls $MODEL | grep 'step[0-9]*-unsharded$' | sort -V | tail -n 6)
        for CKPT in $FINAL_CKPTS; do
            process_model_checkpoint "$MODEL/$CKPT"
        done
    fi
done


#### OLD ####

# for MODEL in "${MODEL_LIST_LADDER[@]}"; do
# # for MODEL in "${MODEL_LIST_CUSTOM[@]}"; do
#     if [[ $MODEL == *"step"* ]]; then
#         # Option 1, use the provided model path
#         MODEL_PATH=$MODEL
#     else
#         # Option 2, find highest model path
#         FINAL_CKPT=$(ls $MODEL | grep 'step[0-9]*-unsharded$' | sort -V | tail -n 1)
#         MODEL_PATH=$MODEL/$FINAL_CKPT
#     fi

#     INPUT_DIR=$MODEL_PATH
#     OUTPUT_DIR=$MODEL_PATH-hf

#     # Skip if OUTPUT_DIR already exists
#     if [[ -d "$OUTPUT_DIR" ]]; then
#         echo "Output directory exists: $OUTPUT_DIR. Skipping..."
#         continue
#     fi

#     echo $INPUT_DIR
#     echo $OUTPUT_DIR

#     # python olmo-repos/OLMo/scripts/convert_olmo2_to_hf.py \
#     #     --input_dir "$INPUT_DIR" \
#     #     --output_dir "$OUTPUT_DIR" \
#     #     --tokenizer_json_path "$TOKENIZER_PATH"
# done