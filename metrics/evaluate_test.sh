#!/bin/sh

python3 -m metrics.evaluate_hasqi --input_csv ../Radio/csv/ak_train_1.csv --output_csv test.csv \
--source_key clean --target_key noisy

