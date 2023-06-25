use std::{ptr::NonNull, marker::PhantomPinned, pin::Pin, cmp::min, num::NonZeroUsize, fmt::Display, error::Error};

use openai_rs::{chat::{ChatMessage, Role, ChatHistoryBuilder}, context::Context as OpenAIContext};
use tiktoken::CoreBPE;

const PROMPT_COMPRESS: &str = "Summarize the chat history precisely and concisely";

type UserAliases = Vec<String>;

#[derive(Debug)]
struct ContextOverrunError {
    max_tokens: usize,
    context_budget: usize,
    history_budget: usize,
    alias_budget: usize
}

impl Display for ContextOverrunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("Context budget overrun. Context ({}), history ({}) and alias ({}) budgets must be at most {} tokens", self.context_budget, self.history_budget, self.alias_budget, self.max_tokens))?;
        Ok(())
    }
}

impl Error for ContextOverrunError {}

impl ContextOverrunError {
    fn new(max_tokens: usize, context_budget: usize, history_budget: usize, alias_budget: usize) -> Self {
        Self {
            max_tokens,
            context_budget,
            history_budget,
            alias_budget
        }
    }
}

pub struct UserList {
    pub users: Vec<UserAliases>,
    _pin: PhantomPinned
}

pub enum User {
    Assistant,
    System,
    User {
        aliases: NonNull<UserAliases>
    }
}

pub struct Message {
    pub sender: User,
    pub message: String
}

impl Message {
    fn new(sender: User, message: String) -> Self {
        Self {
            sender,
            message
        }
    }
    
    fn to_chat_message(&self, user_index: Option<usize>) -> ChatMessage {
        ChatMessage::new(
            match self.sender {
                User::System => Role::System,
                User::Assistant => Role::Assistant,
                User::User { aliases } => Role::User
            },
            self.message,
            if let Some(user_index) = user_index {
                Some(format!("u{user_index}"))
            } else {
                None
            }
        )
    }
}

pub struct Context {
    pub users: Pin<Box<UserList>>,
    messages: Vec<Message>,
    openai_context: OpenAIContext,
    summary: Option<String>,
    max_tokens: usize,
    model: String,
    encoding: CoreBPE,
    summary_budget: usize,
    summary_instruction_budget: usize,
    history_target: usize,
    alias_budget: usize,
}

impl UserList {
    fn new() -> Pin<Box<Self>> {
        Box::pin(Self {
            users: Vec::new(),
            _pin: PhantomPinned
        })
    }

    fn add_user(&mut self) {
        self.users.push(Vec::new());
    }

    fn add_existing_user(&mut self, aliases: UserAliases) {
        self.users.push(aliases);
    }
}

impl Context {
    fn new(max_tokens: NonZeroUsize, model: String, encoding: CoreBPE, openai_context: OpenAIContext, summary_budget: NonZeroUsize, history_target: NonZeroUsize, alias_budget: NonZeroUsize) -> Result<Self, impl Error> {
        let summary_instruction_budget = count_message_tokens(&get_summary_instruction(), &encoding, &model) as usize;
        let summary_budget = summary_budget.get() + count_message_tokens(&get_summary_message(None), &encoding, &model) as usize;
        if history_target.get() + summary_budget + alias_budget.get() + summary_instruction_budget >= max_tokens.get() {
            Err(ContextOverrunError::new(max_tokens.get(), history_target.get(), summary_budget, alias_budget.get()))
        } else {
            Ok(Self {
                users: UserList::new(),
                messages: Vec::new(),
                openai_context,
                summary: None,
                max_tokens: max_tokens.get(),
                model,
                encoding,
                summary_budget: summary_budget,
                summary_instruction_budget,
                history_target: history_target.get(),
                alias_budget: alias_budget.get()
            })
        }
    }

    fn find_user_by_alias(&self, find: NonNull<UserAliases>) -> Option<usize> {
        for (index, user) in self.users.users.iter().enumerate() {
            if unsafe { find.as_ref() == user } {
                return Some(index);
            }
        }
        return None;
    }

    fn find_user(&self, find: &User) -> Option<usize> {
        if let User::User { aliases } = find {
            self.find_user_by_alias(*aliases)
        } else {
            None
        }
    }

    fn update_user_list(&mut self, user: &User) -> Option<usize> {
        if let User::User { aliases } = user {
            Some(if let Some(index) = self.find_user_by_alias(*aliases) {
                index
            } else {
                eprintln!("Attempt to add unregistered user to history! This is probably a bug.");

                let copy = unsafe { (*aliases.as_ptr()).clone() };
                self.users.add_existing_user(copy);
                self.users.users.len() - 1
            })
        } else {
            None
        }
    }

    fn history_token_limit(&self) -> usize {
        self.max_tokens - self.alias_budget - self.summary_budget - self.summary_instruction_budget
    }

    pub async fn add_message(&mut self, message: String, user: User) {
        let user_index = self.update_user_list(&user);
        let total_tokens = self.count_message_tokens();
        let message = Message::new(user, message);

        let message_tokens = count_message_tokens(&message.to_chat_message(user_index), &self.encoding, &self.model);

        if (total_tokens + message_tokens) as usize >= self.history_token_limit() {
            self.compress_history(message_tokens as usize).await;
        }
        
        self.messages.push(message);
    }

    pub async fn add_message_0(&mut self, message: Message) {
        self.add_message(message.message, message.sender).await
    }

    pub async fn generate_response(&self) -> anyhow::Result<Option<Message>> {
        let mut history = self.chat_to_history(None);
        if let Some(ref summary) = self.summary {
            history.insert(0, get_summary_message(Some(summary.clone())));
        }

        let response = self.openai_context.create_chat_completion_sync(
            ChatHistoryBuilder::default()
                .messages(history)
                .max_tokens(self.history_token_limit() as u64)
                .model(self.model.clone())
        ).await?.choices.remove(0).message.content;

        Ok(if response.len() > 0 {
            Some(Message::new(User::Assistant, response))
        } else {
            None
        })
    }

    pub fn get_user_aliases(&self, id: usize) -> Option<&mut UserAliases> {
        if id >= self.users.users.len() {
            None
        } else {
            Some(&mut self.users.users[id])
        }
    }

    pub fn chat_to_history(&self, last_n: Option<usize>) -> Vec<ChatMessage> {
        let last_n = min(if let Some(value) = last_n { value } else { self.messages.len() }, self.messages.len());

        let mut history = Vec::new();

        for (index, msg) in self.messages[self.messages.len() - last_n..self.messages.len()].iter().enumerate() {
            history.push(msg.to_chat_message(if let User::User { aliases } = msg.sender {
                Some(index)
            } else {
                None
            }));
        }

        return history;
    }

    fn count_message_tokens(&self) -> i64 {
        let history = self.chat_to_history(None);
        let mut total = 0i64;

        for ref message in history {
            total += count_message_tokens(message, &self.encoding, &self.model);
        }

        return total;
    }

    async fn compress_history(&mut self, new_tokens: usize) -> anyhow::Result<()> {
        let mut permitted_history_size = if new_tokens > self.history_target {
            0
        } else {
            self.history_target - new_tokens
        };

        let mut skip_count = 0;

        for (index, message) in self.messages.iter().enumerate() {
            let tokens = count_message_tokens(&message.to_chat_message(self.find_user(&message.sender)), &self.encoding, &self.model) as usize;
            if tokens > permitted_history_size {
                break;
            }

            permitted_history_size -= tokens;
            skip_count = index + 1;
        }

        let mut history = self.chat_to_history(Some(self.messages.len() - skip_count));
        history.push(get_summary_instruction());
        self.summary = Some(self.openai_context.create_chat_completion_sync(
            ChatHistoryBuilder::default()
                .max_tokens(self.summary_budget as u64)
                .model(self.model.clone())
                .messages(history)
        ).await?.choices.remove(0).message.content);

        self.messages.drain(skip_count..);

        Ok(())
    }
}

fn get_summary_instruction() -> ChatMessage {
    ChatMessage::new(Role::System, PROMPT_COMPRESS, None)
}
fn get_summary_message(summary: Option<String>) -> ChatMessage {
    ChatMessage::new(Role::System, if let Some(ref message) = summary { message } else { "" }, Some("Context".to_string()))
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