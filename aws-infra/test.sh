sam build --use-container

sam local invoke HelloWorldFunction -e events/event.json
