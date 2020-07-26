sam build --use-container

sam package --s3-bucket landis-cloudformation-template --output-template-file packaged.yaml

sam deploy --template-file ./packaged.yaml --stack-name landis-inference --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND
