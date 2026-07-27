import boto3
from botocore.exceptions import ClientError

def update_task_status(
    task_id: str, 
    status: str, 
    table_name: str = "AgentTasks", 
    additional_attributes: dict = None
) -> bool:
    """
    Updates the task status in DynamoDB and dynamically sets additional attributes.

    :param task_id: The primary key (partition key) of the task.
    :param status: The new status string (e.g., 'planning_complete', 'sync_complete', 'failed').
    :param table_name: The DynamoDB table name.
    :param additional_attributes: Optional dict of key-value pairs to set in DynamoDB.
    :return: True if update succeeded, False otherwise.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    # Use '#st' alias because 'status' is a reserved keyword in DynamoDB
    update_expression = "SET #st = :status_val"
    expression_attribute_names = {"#st": "status"}
    expression_attribute_values = {":status_val": status}

    # Dynamically append additional fields if provided
    if additional_attributes:
        for key, value in additional_attributes.items():
            # Avoid overwriting status if passed again in dict
            if key == "status":
                continue
            
            placeholder_name = f"#{key}"
            placeholder_val = f":{key}_val"
            
            update_expression += f", {placeholder_name} = {placeholder_val}"
            expression_attribute_names[placeholder_name] = key
            expression_attribute_values[placeholder_val] = value

    try:
        table.update_item(
            Key={'id': task_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues="UPDATED_NEW"
        )
        print(f"🔄 DynamoDB Task '{task_id}' status updated to -> '{status}'")
        return True

    except ClientError as e:
        print(f"❌ Failed to update DynamoDB task status: {e.response['Error']['Message']}")
        return False