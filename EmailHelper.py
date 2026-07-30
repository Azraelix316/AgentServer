import boto3
from botocore.exceptions import ClientError

ses_client = boto3.client('ses', region_name='ap-southeast-2')

SENDER_EMAIL = "jaredj13310@gmail.com"
RECIPIENT_EMAIL = "jaredj13310@gmail.com"

def send_task_completion_email(task_id, task_name, status, report_summary=""):
    subject = f"[Task {status.upper()}] {task_name} ({task_id[:8]})"
    
    body_text = f"""
    Task Status Update:
    -------------------
    Task ID: {task_id}
    Task Name: {task_name}
    Status: {status}
    
    Summary / Excerpt:
    {report_summary[:500]}...
    """

    try:
        ses_client.send_email(
            Source=SENDER_EMAIL,
            Destination={'ToAddresses': [RECIPIENT_EMAIL]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body_text}}
            }
        )
        print(f"Completion email sent for task {task_id}")
    except ClientError as e:
        print(f"Failed to send SES email: {e.response['Error']['Message']}")