import json
# import sagemaker
import base64
# from sagemaker.serializers import IdentitySerializer

# for calling endpoint directly
import boto3

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2022-08-28-09-15-46-373'


def lambda_handler(event, context):
    image_data = event['body']['image_data']

    # Decode the image data
    image = base64.b64decode(image_data)

    # print(image)

    # calling the endpoint directly, because the packaging does not work
    endpoint = ENDPOINT
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='image/png', Body=image)

    # print(response)

    predictions = json.loads(response['Body'].read().decode())

    # print(predictions)

    """
    # Decode the image data
    image = base64.b64decode(image_data)

    # Instantiate a Predictor
    predictor = sagemaker.Predictor(ENDPOINT, sagemaker_session=sagemaker.Session())

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    """

    event["inferences"] = predictions

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }