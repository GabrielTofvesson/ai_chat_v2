use std::{error::Error};

use openai_rs::{chat::{ChatMessage, Role, ChatHistoryBuilder}, context::Context, edits::EditRequestBuilder};
use tiktoken::{CoreBPE, model::{model_cl100k_base, cl100k_base}};

#[derive(Debug, Clone)]
pub struct ChatContextError<'l> {
    reason: &'l str
}

impl std::fmt::Display for ChatContextError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl Error for ChatContextError<'_> {
    
}

#[derive(Clone)]
pub enum MessageType {
    AssistantMessage,
    UserMessage {
        sender: UserAlias,
    }
}

#[derive(Clone)]
pub struct MetaChatMessage {
    pub chat_message: ChatMessage,
    pub message_type: MessageType,
}

#[derive(Clone)]
pub struct UserAlias {
    id: u16,
    names: Vec<String>,
}

pub struct ChatContext {
    model: String,
    encoding: CoreBPE,
    max_tokens: i64,
    api_context: Context,
    history: Vec<MetaChatMessage>,
    context: Option<String>,
    user_aliases: Vec<UserAlias>,
}

impl ChatContext {
    pub async fn new(model: String, api_key: String) -> anyhow::Result<Self> {
        Ok(Self {
            encoding: get_model(&model).await.ok_or(ChatContextError { reason: "Couldn't get model encoding" })?,
            max_tokens: get_max_tokens(&model).ok_or(ChatContextError { reason: "Couldn't get max tokens for model" })?,
            api_context: Context::new(api_key.to_string()),
            history: Vec::new(),
            context: None,
            model,
            user_aliases: Vec::new()
        })
    }

    pub async fn send_message(&mut self, message: MetaChatMessage) -> MetaChatMessage {
        self.history.push(message);
        let tpm = get_tokens_per_message(&self.model).unwrap();
        let message_token_count = count_tokens(&self.history, &self.encoding, &self.model) + tpm;
        if message_token_count >= self.max_tokens - tpm {
            panic!("Message history exceeds token limit! No new message can be generated.");
        }

        // Compute maximum number of tokens to generate
        let max_tokens = self.max_tokens - message_token_count - tpm - 1;


        let completion = self.api_context
            .create_chat_completion_sync(
                ChatHistoryBuilder::default()
                    .temperature(0.3) // Model suffers from excessive hallucination. TODO: fine-tune temperature
                    .messages(self.history.iter().map(|message| message.chat_message.clone()).collect::<Vec<ChatMessage>>())
                    .max_tokens(max_tokens as u64)
                    .model(&self.model),
            )
            .await;
        assert!(
            completion.is_ok(),
            "Could not create completion: {}",
            completion.unwrap_err()
        );

        let mut result = completion.unwrap();
        assert!(result.choices.len() == 1, "No completion found");
        return MetaChatMessage {
            chat_message: result.choices.pop().unwrap().message,
            message_type: MessageType::AssistantMessage
        };
    }

    async fn update_aliases(&self, instruction: &str, aliases: &mut Vec<UserAlias>, message_context: &[MetaChatMessage], context_count: usize) -> anyhow::Result<()> {
        if message_context.len() < context_count {
            return Ok(());
        }
        let latest = &message_context[message_context.len() - 1];
        if let MessageType::UserMessage { ref sender } = latest.message_type {
            let mut alias_prompt = String::new();
    
            for alias in aliases {
                alias_prompt.push_str(&format!("u{}:", alias.id));
    
                for name in &alias.names {
                    alias_prompt.push_str(&format!(" {name},"));
                }
    
                if alias.names.len() > 0 {
                    alias_prompt.pop();
                }
            }
    
            let mut instruction = String::new();
            instruction.push_str("Update the list of user aliases based on the chat message:");
            instruction.push_str(&format!("\nu{}: \"{}\"", sender.id, latest.chat_message.content));
    
            let edit = self.api_context.create_edit(
                EditRequestBuilder::default()
                    .input(alias_prompt)
                    .instruction(format!(""))
                    .build()?
            );
        }

        return Ok(());
    }

    pub fn get_history(&mut self) -> &mut Vec<MetaChatMessage> {
        &mut self.history
    }
}



async fn get_model(model: &str) -> Option<CoreBPE> {
    return match model {
        "gpt-4" | "gpt-4-32k" | "gpt-3.5-turbo" | "text-embedding-ada-002" => {
            let model = model_cl100k_base().await;
            assert!(model.is_ok(), "Could not download model (model_cl100k_base): {:?}", model);

            let model = cl100k_base(model.unwrap());
            assert!(model.is_ok(), "Could not load model (cl100k_base): {:?}", model.err().unwrap());
            
            return Some(model.unwrap());
        }
        _ => None
    }
}

fn get_tokens_per_message(model: &str) -> Option<i64> {
    match model {
        "gpt-4" | "gpt-4-32k" => Some(3),
        "gpt-3.5-turbo" => Some(4),
        _ => None
    }
}

fn get_tokens_per_name(model: &str) -> Option<i64> {
    match model {
        "gpt-4" | "gpt-4-32k" => Some(1),
        "gpt-3.5-turbo" => Some(-1),
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

fn count_message_tokens(message: &ChatMessage, encoding: &CoreBPE, model: &str) -> i64 {
    let tpm = get_tokens_per_message(model).expect("Unknown tokens-per-message value");
    let tpn = get_tokens_per_name(model).expect("Unknown tokens-per-name value");

    return tpm + encoding.encode_ordinary(&message.content).len() as i64 + encoding.encode_ordinary(role_str(&message.role)).len() as i64 + if let Some(ref name) = message.name {
        tpn + encoding.encode_ordinary(name).len() as i64
    } else { 0i64 };
}

fn count_tokens(history: &Vec<MetaChatMessage>, encoding: &CoreBPE, model: &str) -> i64 {
    let mut count = 0i64;
    let tpm = get_tokens_per_message(model).expect("Unknown tokens-per-message value");
    let tpn = get_tokens_per_name(model).expect("Unknown tokens-per-name value");
    for entry in history {
        count += tpm + encoding.encode_ordinary(&entry.chat_message.content).len() as i64 + encoding.encode_ordinary(role_str(&entry.chat_message.role)).len() as i64;

        if entry.chat_message.name.is_some() {
            count += tpn + encoding.encode_ordinary(entry.chat_message.name.as_ref().unwrap()).len() as i64;
        }
    }
    return count;
}

fn get_max_tokens(model: &str) -> Option<i64> {
    match model {
        "gpt-4" => Some(8192),
        "gpt-4-32k" => Some(32768),
        "gpt-3.5-turbo" => Some(4096),
        "code-davinci-002" => Some(8001),
        _ => None
    }
}