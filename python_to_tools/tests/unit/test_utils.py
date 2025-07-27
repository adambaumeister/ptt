
def test_prompt_template_to_messages():
    from python_to_tools.utils import prompt_template_to_convo
    print(prompt_template_to_convo("single_task_agent.j2"))