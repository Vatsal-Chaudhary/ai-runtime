use std::{collections::HashMap, sync::Arc, time::Duration};

use futures_util::future::BoxFuture;
use serde_json::{Map, Value, json};
use thiserror::Error;
use tokio::time;

use crate::llm::ToolSpec;

pub type Result<T> = std::result::Result<T, ToolError>;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool already registered: {name}")]
    DuplicateTool { name: String },

    #[error("tool not found: {name}")]
    NotFound { name: String },

    #[error("invalid arguments for tool {tool}: {message}")]
    InvalidArguments { tool: String, message: String },

    #[error("tool timed out: {name}")]
    Timeout { name: String },

    #[error("tool {name} failed: {message}")]
    CallFailed { name: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolMetadata {
    pub read_only: bool,
    pub idempotent: bool,
    pub timeout: Duration,
    pub mutates_resource: Option<String>,
}

impl ToolMetadata {
    pub fn read_only(timeout: Duration) -> Self {
        Self {
            read_only: true,
            idempotent: true,
            timeout,
            mutates_resource: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub metadata: ToolMetadata,
}

impl ToolDefinition {
    pub fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    fn call(&self, args: Value) -> BoxFuture<'static, Result<Value>>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T) -> Result<()>
    where
        T: Tool + 'static,
    {
        let definition = tool.definition();
        if self.tools.contains_key(&definition.name) {
            return Err(ToolError::DuplicateTool {
                name: definition.name,
            });
        }

        self.tools.insert(definition.name, Arc::new(tool));
        Ok(())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = self
            .tools
            .values()
            .map(|tool| tool.definition())
            .collect::<Vec<_>>();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    pub fn definition(&self, name: &str) -> Option<ToolDefinition> {
        self.tools.get(name).map(|tool| tool.definition())
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.definitions()
            .into_iter()
            .map(|definition| definition.spec())
            .collect()
    }

    pub async fn call(&self, name: &str, args: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .cloned()
            .ok_or_else(|| ToolError::NotFound {
                name: name.to_string(),
            })?;

        let definition = tool.definition();
        validate_arguments(&definition.name, &definition.parameters, &args)?;

        match time::timeout(definition.metadata.timeout, tool.call(args)).await {
            Ok(result) => result,
            Err(_) => Err(ToolError::Timeout {
                name: definition.name,
            }),
        }
    }
}

fn validate_arguments(tool: &str, schema: &Value, args: &Value) -> Result<()> {
    validate_value(tool, schema, args, "$")
}

fn validate_value(tool: &str, schema: &Value, value: &Value, path: &str) -> Result<()> {
    if let Some(expected_type) = schema.get("type").and_then(Value::as_str) {
        let type_matches = match expected_type {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => true,
        };

        if !type_matches {
            return invalid(tool, format!("{path} must be {expected_type}"));
        }
    }

    if let Some(options) = schema.get("enum").and_then(Value::as_array)
        && !options.iter().any(|option| option == value)
    {
        return invalid(tool, format!("{path} is not one of the allowed values"));
    }

    if schema.get("type").and_then(Value::as_str) == Some("object") {
        validate_object(tool, schema, value, path)?;
    }

    Ok(())
}

fn validate_object(tool: &str, schema: &Value, value: &Value, path: &str) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| ToolError::InvalidArguments {
            tool: tool.to_string(),
            message: format!("{path} must be object"),
        })?;

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for field in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(field) {
                return invalid(tool, format!("{path}.{field} is required"));
            }
        }
    }

    let empty_properties = Map::new();
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or(&empty_properties);

    if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
        for key in object.keys() {
            if !properties.contains_key(key) {
                return invalid(tool, format!("{path}.{key} is not allowed"));
            }
        }
    }

    for (field, field_schema) in properties {
        if let Some(field_value) = object.get(field) {
            validate_value(tool, field_schema, field_value, &format!("{path}.{field}"))?;
        }
    }

    Ok(())
}

fn invalid<T>(tool: &str, message: String) -> Result<T> {
    Err(ToolError::InvalidArguments {
        tool: tool.to_string(),
        message,
    })
}

#[derive(Debug, Clone)]
pub struct CalculatorTool {
    timeout: Duration,
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(250),
        }
    }
}

impl Tool for CalculatorTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".to_string(),
            description: "Performs one arithmetic operation on two numbers.".to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["operation", "a", "b"],
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "a": { "type": "number" },
                    "b": { "type": "number" }
                }
            }),
            metadata: ToolMetadata::read_only(self.timeout),
        }
    }

    fn call(&self, args: Value) -> BoxFuture<'static, Result<Value>> {
        Box::pin(async move {
            let operation = required_str("calculator", &args, "operation")?;
            let a = required_f64("calculator", &args, "a")?;
            let b = required_f64("calculator", &args, "b")?;

            let result = match operation {
                "add" => a + b,
                "subtract" => a - b,
                "multiply" => a * b,
                "divide" if b == 0.0 => {
                    return Err(ToolError::CallFailed {
                        name: "calculator".to_string(),
                        message: "division by zero".to_string(),
                    });
                }
                "divide" => a / b,
                _ => {
                    return Err(ToolError::InvalidArguments {
                        tool: "calculator".to_string(),
                        message: "operation is not supported".to_string(),
                    });
                }
            };

            Ok(json!({ "result": result }))
        })
    }
}

#[derive(Debug, Clone)]
pub struct MockCrmLookupTool {
    timeout: Duration,
    customers: Arc<HashMap<String, Value>>,
}

impl MockCrmLookupTool {
    pub fn new(customers: HashMap<String, Value>) -> Self {
        Self {
            timeout: Duration::from_millis(250),
            customers: Arc::new(customers),
        }
    }
}

impl Tool for MockCrmLookupTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "mock_crm_lookup".to_string(),
            description: "Looks up a mock CRM customer by customer_id.".to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["customer_id"],
                "properties": {
                    "customer_id": { "type": "string" }
                }
            }),
            metadata: ToolMetadata::read_only(self.timeout),
        }
    }

    fn call(&self, args: Value) -> BoxFuture<'static, Result<Value>> {
        let customers = Arc::clone(&self.customers);
        Box::pin(async move {
            let customer_id = required_str("mock_crm_lookup", &args, "customer_id")?;
            customers
                .get(customer_id)
                .cloned()
                .ok_or_else(|| ToolError::CallFailed {
                    name: "mock_crm_lookup".to_string(),
                    message: format!("customer {customer_id} was not found"),
                })
        })
    }
}

fn required_str<'a>(tool: &str, args: &'a Value, field: &str) -> Result<&'a str> {
    args.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidArguments {
            tool: tool.to_string(),
            message: format!("$.{field} must be string"),
        })
}

fn required_f64(tool: &str, args: &Value, field: &str) -> Result<f64> {
    args.get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| ToolError::InvalidArguments {
            tool: tool.to_string(),
            message: format!("$.{field} must be number"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn registry_dispatches_registered_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register(CalculatorTool::default())
            .expect("tool should register");

        let result = registry
            .call(
                "calculator",
                json!({
                    "operation": "multiply",
                    "a": 6,
                    "b": 7
                }),
            )
            .await
            .expect("tool call should succeed");

        assert_eq!(result, json!({ "result": 42.0 }));
    }

    #[tokio::test]
    async fn registry_rejects_invalid_arguments_before_dispatch() {
        let mut registry = ToolRegistry::new();
        registry
            .register(CalculatorTool::default())
            .expect("tool should register");

        let err = registry
            .call(
                "calculator",
                json!({
                    "operation": "multiply",
                    "a": 6,
                    "extra": true
                }),
            )
            .await
            .expect_err("invalid args should fail");

        assert!(matches!(err, ToolError::InvalidArguments { .. }));
    }

    #[tokio::test]
    async fn registry_returns_timeout_for_slow_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(SlowTool).expect("tool should register");

        let err = registry
            .call("slow", json!({}))
            .await
            .expect_err("slow tool should time out");

        assert!(matches!(err, ToolError::Timeout { name } if name == "slow"));
    }

    struct SlowTool;

    impl Tool for SlowTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "slow".to_string(),
                description: "Sleeps longer than its timeout.".to_string(),
                parameters: json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                }),
                metadata: ToolMetadata::read_only(Duration::from_millis(10)),
            }
        }

        fn call(&self, _args: Value) -> BoxFuture<'static, Result<Value>> {
            Box::pin(async {
                time::sleep(Duration::from_secs(30)).await;
                Ok(json!({ "ok": true }))
            })
        }
    }
}
