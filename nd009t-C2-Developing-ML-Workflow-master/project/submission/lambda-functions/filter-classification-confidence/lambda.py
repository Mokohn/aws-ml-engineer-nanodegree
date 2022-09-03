THRESHOLD = .90


def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event['inferences']

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = [inference > THRESHOLD for inference in inferences]
    print(f'Meets treshold: {meets_threshold}')

    if True in meets_threshold:
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    #if meets_threshold:
        pass
    else:
        raise Exception(f"THRESHOLD_CONFIDENCE_NOT_MET, INFERENCES {inferences}")
        
    return {
        'statusCode': 200,
        'body': json.dumps(event)
        }