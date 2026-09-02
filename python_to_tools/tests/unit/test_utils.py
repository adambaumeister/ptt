from python_to_tools.ai.generic_models import MessageRoleEnum
from python_to_tools.utils import JinjaConvoLoader


def test_prompt_template_to_messages():
    from python_to_tools.utils import JinjaConvoLoader
    from python_to_tools.ptt import MemoryContext
    context = MemoryContext()
    context.add_message("Example ASsistnat message 1", role=MessageRoleEnum.assistant)
    loader = JinjaConvoLoader("single_task_agent.j2")
    result = loader.to_convo(last_input="Example user message", context=context)
    print(result.model_dump_json(indent=4))