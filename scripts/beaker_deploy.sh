DATA_DIR=/root/ai2/metaeval/data # data to copy into Dockerfile

cd ~/ai2/metaeval/olmo-repos/oe-eval-internal && \
rm -rf data && \
cp -r $DATA_DIR data && \
docker build -t oe-eval-metaeval . -f Dockerfile_beaker  && \
beaker image delete davidh/oe-eval-metaeval && \
beaker image create --name oe-eval-metaeval --workspace ai2/davidh oe-eval-metaeval &&
/root/ai2/metaeval/test_eval_beaker.sh