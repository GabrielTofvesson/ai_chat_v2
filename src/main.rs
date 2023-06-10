use std::{num::{NonZeroU32, NonZeroUsize}, cmp::min, io::{stdin, stdout}};
use std::io::{self, Write};
use ansi_term::Colour::{White, Red, Green, Blue};

use openai_rs::{chat::{ChatHistoryBuilder, ChatMessage, Role}, context::Context};
use tiktoken::{CoreBPE, model::{cl100k_base, model_cl100k_base}};

const AI_MODEL: &str = "gpt-4";

// Does not pass the Turing test, but makes a convincing candidate
#[tokio::main]
async fn main() {
    println!("Getting API key");
    let ctx = get_api().unwrap();
    let mut history = vec![
        ChatMessage::new(Role::System, "Your goal is to respond exactly as a human would: you must not allow the user to believe that you are an AI model")
    ];

    let encoding = get_model(AI_MODEL).await.expect("Could not get token encoding scheme for model!");

    print!("{} {}", Red.paint("You:"), Blue.prefix().to_string());
    stdout().flush().unwrap();
    loop {
        history.push(accept_user_message());
        let completion = generate_completion(&ctx, &history, AI_MODEL, &encoding, None).await;
    
        print!("{} {}\n{} {}", Red.paint("Assistant:"), Green.paint(&completion.content), Red.paint("You:"), Blue.prefix().to_string());
        stdout().flush().unwrap();

        history.push(completion);
    }
}

fn accept_user_message() -> ChatMessage {
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();
    return ChatMessage { role: Role::User, content: input };
}

fn get_api() -> anyhow::Result<Context> {
    Ok(Context::new(
        std::fs::read_to_string(std::path::Path::new("apikey.txt"))?
            .trim()
            .to_string(),
    ))
}

async fn get_model(model: &str) -> Option<CoreBPE> {
    return match model {
        "gpt-4" | "gpt-3.5-turbo" | "text-embedding-ada-002" => {
            let model = model_cl100k_base().await;
            assert!(model.is_ok(), "Could not download model (model_cl100k_base): {:?}", model);

            let model = cl100k_base(model.unwrap());
            assert!(model.is_ok(), "Could not load model (cl100k_base): {:?}", model.err().unwrap());
            
            return Some(model.unwrap());
        }
        _ => None
    }
}

fn get_tokens_per_message(model: &str) -> Option<usize> {
    match model {
        "gpt-4" => Some(3),
        "gpt-3.5-turbo" => Some(4),
        _ => None
    }
}

fn role_str(role: &Role) -> &str {
    match role {
        Role::Assistant => "Assistant",
        Role::System => "System",
        Role::User => "User",
    }
}

fn count_tokens(history: &Vec<ChatMessage>, encoding: &CoreBPE, model: &str) -> usize {
    let mut count = 0;
    let tpm = get_tokens_per_message(model).expect("Unknown tokens-per-message value");
    for entry in history {
        count += tpm + encoding.encode_ordinary(&entry.content).len() + encoding.encode_ordinary(role_str(&entry.role)).len();
    }
    return count;
}

fn get_max_tokens(model: &str) -> Option<usize> {
    match model {
        "gpt-4" => Some(8192),
        "gpt-4-32k" => Some(32768),
        "gpt-3.5-turbo" => Some(4096),
        "code-davinci-002" => Some(8001),
        _ => None
    }
}

async fn generate_completion(ctx: &Context, history: &Vec<ChatMessage>, model: &str, encoding: &CoreBPE, token_limit: Option<NonZeroUsize>) -> ChatMessage {
    let message_token_count = count_tokens(history, encoding, model);
    let abs_max = get_max_tokens(model).expect("Undefined maximum token count for model!");

    if message_token_count >= abs_max - get_tokens_per_message(model).unwrap() {
        panic!("Message history exceeds token limit! No new message can be generated.");
    }

    // Compute maximum number of tokens to generate
    let max_tokens = match token_limit {
        Some(lim) => min(abs_max - message_token_count, lim.get()),
        _ => abs_max - message_token_count
    };

    
    let completion = ctx
        .create_chat_completion_sync(
            ChatHistoryBuilder::default()
                .messages(history.clone())
                .model(model),
        )
        .await;
    assert!(
        completion.is_ok(),
        "Could not create completion: {}",
        completion.unwrap_err()
    );

    let mut result = completion.unwrap();
    assert!(result.choices.len() == 1, "No completion found");

    return result.choices.pop().unwrap().message;
}