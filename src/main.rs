use openai_rs::{chat::{ChatHistoryBuilder, ChatMessage, Role}, context::Context};

#[tokio::main]
async fn main() {
    println!("Getting API key");
    let ctx = get_api().unwrap();

    println!("Generating completion...");
    let completion = ctx
        .create_chat_completion_sync(
            ChatHistoryBuilder::default()
                .messages(vec![ChatMessage::new(Role::User, "Who are you?")])
                .model("gpt-4"),
        )
        .await;
    assert!(
        completion.is_ok(),
        "Could not create completion: {}",
        completion.unwrap_err()
    );

    let result = completion.unwrap();
    assert!(result.choices.len() == 1, "No completion found");
    println!("Got completion: {:?}", result.choices[0].message);
}

fn get_api() -> anyhow::Result<Context> {
    Ok(Context::new(
        std::fs::read_to_string(std::path::Path::new("apikey.txt"))?
            .trim()
            .to_string(),
    ))
}
