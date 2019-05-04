mkdir -p model_zoo_tests/nitro_model_zoo
aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300 s3 sync s3://nitro_model_zoo/MT-bi-en_Var-zh_CN-v106-20190114-d299c44b0 nitro_model_zoo/
