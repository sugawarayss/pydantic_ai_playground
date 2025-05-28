import asyncio

import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel("o4-mini", provider=OpenAIProvider(api_key="")) # TODO: OpenAI APIキーを入力する必要あり
agent: Agent = Agent(
    model=model,
    system_prompt="あなたは優秀なプログラマです。ユーザからの質問に対して、正確で有用な回答を提供してください。語尾に「アル」と付けてください。",
)

# チャット履歴をセッションステートに初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.write(message["content"])

# チャット入力欄
user_msg: str | None = st.chat_input(placeholder="メッセージを入力してください")

async def invoke(user_msg: str):
    async with agent.run_stream(user_prompt=user_msg) as stream:
        async for message in stream.stream_text(delta=True):
            yield message

if user_msg:
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # ユーザーメッセージを表示
    with st.chat_message(name="user"):
        st.write(user_msg)

    # アシスタントの応答用のプレースホルダー
    with st.chat_message(name="assistant"):
        message_placeholder = st.empty()
        # full_response = ""

        # asyncioを使ってストリーミング応答を処理
        async def process_response():
            response_text = ""
            async for chunk in invoke(user_msg):
                response_text += chunk
                message_placeholder.write(response_text)
            return response_text

        # asyncio関数をStreamlitで実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        full_response = loop.run_until_complete(process_response()) or ""

    # 完成した応答をセッションステートに保存
    st.session_state.messages.append({"role": "assistant", "content": full_response})