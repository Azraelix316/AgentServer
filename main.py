from KaggleHelper import KaggleHelper
if __name__ == "__main__":
    AWS_PROFILE = "test_only"
    boto3.setup_default_session(profile_name=AWS_PROFILE)
    helper = KaggleHelper(
        kaggle_username="my_kaggle_user",
        webhook_url=" https://0flgutlom2.execute-api.ap-southeast-2.amazonaws.com/updateTasks"
    )

    agent_code = """
import pandas as pd
print("Performing analysis...")
# Your agent's logic here
"""

    slug = helper.prepare_and_push(
        task_name="task-101",
        task_id="task-101",
        agent_python_code=agent_code,
        database_link="kaggle:zillow/zecon"
    )
    print(f"Task pushed under slug: {slug}")