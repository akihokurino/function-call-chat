type Message = {
    role: "user" | "assistant";
    content: string;
};

export const chatCompletionsAPI = async (
    messages: Message[],
    received: (text: string) => void,
    finish: () => void
) => {
    const response = await fetch("http://localhost:8080/chat_completion", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({messages}),
    });

    if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (data.response) {
        received(data.response);
    } else {
        received("エラー: レスポンスが不正です");
    }

    finish();
};