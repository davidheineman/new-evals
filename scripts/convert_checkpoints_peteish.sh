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

MODEL_LIST_13B=(
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step0"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step4000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step8000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step12000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step16000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step20000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step24000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step28000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step32000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step36000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step40000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step44000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step48000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step52000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step56000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step60000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step64000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step68000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step72000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step76000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step80000"
    "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step84000"

    # downloaded from aws, still needs conversion
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step88000-sharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step92000-sharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step96000-sharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step100000-sharded"
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
    # "/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848"
    # "/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-10xC/step162694-unsharded"

    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/c4-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/c4-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/c4-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/c4-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/DCLM-baseline-4M-5xC/step5745-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/DCLM-baseline-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/DCLM-baseline-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/DCLM-baseline-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw2-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw2-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw2-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw2-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw3-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw3-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw3-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw3-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top10-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top10-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top10-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top10-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top3-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top3-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top3-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_fw_top3-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-50p-DCLM-baseline-50p-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-50p-DCLM-baseline-50p-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-50p-DCLM-baseline-50p-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-50p-DCLM-baseline-50p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-75p-DCLM-baseline-25p-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-75p-DCLM-baseline-25p-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-75p-DCLM-baseline-25p-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-75p-DCLM-baseline-25p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma-v1-6-and-sources-baseline-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma-v1-6-and-sources-baseline-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma-v1-6-and-sources-baseline-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma-v1-6-and-sources-baseline-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top10p-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top10p-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top10p-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top10p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top20p-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top20p-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top20p-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_eli5_oh_top20p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_og_eli5_oh_top10p-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_og_eli5_oh_top10p-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_og_eli5_oh_top10p-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_og_eli5_oh_top10p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_tulu_qc_top10-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_tulu_qc_top10-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_tulu_qc_top10-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/falcon_and_cc_tulu_qc_top10-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/fineweb_edu_dedup-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/fineweb_edu_dedup-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/fineweb_edu_dedup-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/fineweb_edu_dedup-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_code-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_code-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_code-60M-5xC/step29062-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_code-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_flan-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_flan-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_flan-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_flan-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_math_no_code-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_math_no_code-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_math_no_code-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_math_no_code-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_reddit-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_reddit-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_reddit-60M-5xC/step29052-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_reddit-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-4M-5xC/step5725-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-20M-5xC/step14594-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-60M-5xC/step29062-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-90M-5xC/step29901-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-4M-5xC/step5735-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-20M-5xC/step14584-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-60M-5xC/step29042-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-90M-5xC/step29901-unsharded"
)

MODEL_LIST_1B=(
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1610000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1620000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1630000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1640000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1650000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1660000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1670000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1680000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1690000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1700000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1710000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1720000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1730000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1740000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1750000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1760000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1770000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1780000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1790000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1800000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1810000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1820000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1830000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1840000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1850000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1860000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1870000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1880000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1890000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1900000-unsharded"
    "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1907359-unsharded"
)

SEED_RUNS=(
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-2435-high-eval-interval-1B-5xC/step81342"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-1029-1B-5xC/step81342"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-01345-1B-5xC/step75000"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-10294-1B-5xC/step17250"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-23095-1B-5xC/step81342"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-39240-1B-5xC/step81342"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-59430-1B-5xC/step81342"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-60439-1B-5xC/step81342"
    "/oe-eval-default/ai2-llm/checkpoints/davidh/seed/OLMo2-data-seed-89632-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-data-seed-28530-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-data-seed-40593-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-data-seed-59602-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-1029-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-1304-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-2435-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-3004-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-3043-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-4921-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-4932-1B-5xC/step51250"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-5093-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-5730-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-5794-1B-5xC/step81342"
    "/oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/OLMo2-9258-1B-5xC/step81342"
)

# Dolma 2 Tokenizer
TOKENIZER_PATH=/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/latest/tokenizer.json

# Consistent ranking tokenizer
# TOKENIZER_PATH=/oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/gpt-neox-olmo-dolma-v1_5.json

# We instead need to change oe-training-default -> oe-eval-default
# /oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr

##################################################
# # These are the new data mixes
# #   they say "-unsharded-hf" but are actually just sharded
# # It's unclear which is the not seed run
# # Need to convert both 1B and 150M
# # Also make sure to take the actual final checkpoint, not the one after the end of training (should be step69369, or step63589??)
# # tree -L 1 /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb
# c4-1B-5xC
# DCLM-baseline-1B-5xC
# dclm_ft7percentile_fw2-1B-5xC
# dclm_ft7percentile_fw3-1B-5xC
# dclm_fw_top10-1B-5xC
# dclm_fw_top3-1B-5xC
# dolma17-1B-5xC
# dolma17-25p-DCLM-baseline-75p-1B-5xC
# dolma17-50p-DCLM-baseline-50p-1B-5xC
# dolma17-75p-DCLM-baseline-25p-1B-5xC
# dolma-v1-6-and-sources-baseline-1B-5xC
# falcon-1B-5xC
# falcon_and_cc-1B-5xC
# falcon_and_cc_eli5_oh_top10p-1B-5xC
# falcon_and_cc_eli5_oh_top20p-1B-5xC
# falcon_and_cc_og_eli5_oh_top10p-1B-5xC
# falcon_and_cc_tulu_qc_top10-1B-5xC
# fineweb_edu_dedup-1B-5xC
# no_code-1B-5xC
# no_flan-1B-5xC
# no_math_no_code-1B-5xC
# no_reddit-1B-5xC
# pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-1B-5xC
# pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-1B-5xC
# prox_fineweb_pro-1B-5xC
##################################################


### 13B processer

# process_model_checkpoint() {
#     INPUT_DIR=$1
#     OUTPUT_DIR=$2

#     # Remove -sharded from output dir (for converting AWS checkpoints)
#     OUTPUT_DIR="${OUTPUT_DIR/-sharded/}"

#     # Skip if OUTPUT_DIR already exists
#     if [[ -d "$OUTPUT_DIR" ]]; then
#         echo "Output directory exists: $OUTPUT_DIR. Skipping..."
#         return
#     fi

#     echo $INPUT_DIR
#     echo $OUTPUT_DIR

#     python olmo-repos/OLMo/scripts/unshard.py "$INPUT_DIR" "$OUTPUT_DIR-unsharded"

#     python olmo-repos/OLMo/scripts/convert_olmo2_to_hf.py \
#         --input_dir "$OUTPUT_DIR-unsharded" \
#         --output_dir "$OUTPUT_DIR" \
#         --tokenizer_json_path "$TOKENIZER_PATH"
# }

# for MODEL in "${MODEL_LIST_13B[@]}"; do
# # for MODEL in "${MODEL_LIST_CUSTOM[@]}"; do
#     if [[ $MODEL == *"step"* ]]; then
#         # Replace oe-training-default with oe-eval-default in the model path
#         OUTPUT_DIR=$(echo "$MODEL" | sed 's/oe-training-default/oe-eval-default/')
#         # Directly process the provided model path
#         process_model_checkpoint "$MODEL" "$OUTPUT_DIR"
#     else
#         # Replace oe-training-default with oe-eval-default in the model path
#         MODEL=$(echo "$MODEL" | sed 's/oe-training-default/oe-eval-default/')
#         # Process only the last model checkpoint
#         FINAL_CKPT=$(ls $MODEL | grep 'step[0-9]*-unsharded$' | sort -V | tail -n 1)
#         process_model_checkpoint "$MODEL/$FINAL_CKPT" "$OUTPUT_DIR/$FINAL_CKPT"
#     fi
# done


### Move checkpoints
# /oe-training-default/ai2-llm/checkpoints/davidh/ladder/checkpoints/
# /oe-eval-default/ai2-llm/checkpoints/davidh/seed/

### Original processor

# # Unshard a folder of checkpoints
# python olmo-repos/OLMo/scripts/storage_cleaner.py unshard \
#     "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1" \
#     "/oe-eval-default/ai2-llm/checkpoints/OLMo-small/peteish1"


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

    # Unsharded -> HF (OLMo 1.5)
    # python olmo-repos/OLMo/scripts/convert_olmo_to_hf_new.py \
    #     --input_dir "$INPUT_DIR" \
    #     --output_dir "$OUTPUT_DIR" \
    #     --tokenizer_json_path "$TOKENIZER_PATH"

    # Unsharded -> HF (OLMo 2)
    # python olmo-repos/OLMo/scripts/convert_olmo2_to_hf.py \
    #     --input_dir "$INPUT_DIR" \
    #     --output_dir "$OUTPUT_DIR" \
    #     --tokenizer_json_path "$TOKENIZER_PATH"

    # OLMo core v2
    olmo-cookbook-eval convert \
        "$INPUT_DIR" \
        -t olmo-core-v2 \
        --huggingface-tokenizer allenai/OLMo-2-1124-7B
}

# for MODEL in "${MODEL_LIST_LADDER[@]}"; do
# for MODEL in "${MODEL_LIST_CUSTOM[@]}"; do
# for MODEL in "${MODEL_LIST_1B[@]}"; do
for MODEL in "${SEED_RUNS[@]}"; do
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