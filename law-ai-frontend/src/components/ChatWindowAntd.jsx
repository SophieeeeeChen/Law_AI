import React, { useEffect, useMemo, useState, useRef } from "react";
import {
  Layout,
  Input,
  Button,
  Card,
  List,
  Typography,
  Upload,
  message,
  Spin,
  Space,
  Divider,
  Tag,
  Collapse,
  Radio,
  Row,
  Col,
  Avatar,
} from "antd";
import {
  SendOutlined,
  UploadOutlined,
  RobotOutlined,
  UserOutlined,
  FileTextOutlined,
  ReloadOutlined,
  QuestionCircleOutlined,
} from "@ant-design/icons";
import "antd/dist/reset.css";

const { Sider, Content } = Layout;
const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function ChatWindowAntd() {
  const [messages, setMessages] = useState([
    { sender: "AI", text: "Hello! Upload a case file and ask me anything about family law." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [devUserId, setDevUserId] = useState("dev-user");
  const [pendingQuestions, setPendingQuestions] = useState([]);
  const [pendingMissingFields, setPendingMissingFields] = useState([]);
  const [clarificationAnswers, setClarificationAnswers] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState("auto");
  const [uploading, setUploading] = useState(false);

  const [sessionId] = useState(() => crypto.randomUUID());
  const messagesEndRef = useRef(null);

  const requestHeaders = useMemo(() => {
    if (!devUserId) return {};
    return { "X-DEV-USER": devUserId };
  }, [devUserId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const resetChat = () => {
    setSelectedCaseId(null);
    setMessages([{ sender: "AI", text: "Hello! Upload a case file and ask me anything about family law." }]);
    setPendingQuestions([]);
    setPendingMissingFields([]);
    setClarificationAnswers([]);
    message.success("Chat reset successfully");
  };

  const loadHistory = async (userId) => {
    if (!userId) return;
    try {
      const response = await fetch(`${API_BASE_URL}/history/${userId}`, {
        headers: requestHeaders,
      });
      const data = await response.json();
      setHistory(data.cases || []);
    } catch (err) {
      console.error(err);
      message.error("Failed to load history");
    }
  };

  useEffect(() => {
    loadHistory(devUserId);
    setPendingQuestions([]);
    setPendingMissingFields([]);
    setClarificationAnswers([]);
  }, [devUserId]);

  const loadCaseIntoChat = (caseItem) => {
    setSelectedCaseId(caseItem.case_id);
    setPendingQuestions([]);
    setPendingMissingFields([]);
    setClarificationAnswers([]);
    const caseMessages = (caseItem.qa || []).flatMap((qa) => [
      { sender: "User", text: qa.question },
      { sender: "AI", text: qa.answer },
    ]);
    setMessages([
      { sender: "AI", text: `Loaded case: ${caseItem.filename}` },
      ...caseMessages,
    ]);
    message.success(`Loaded case: ${caseItem.filename}`);
  };

  const startClarification = (questions, missingFields) => {
    setPendingQuestions(questions);
    setPendingMissingFields(missingFields || []);
    setClarificationAnswers(questions.map(() => ""));
    setMessages((msgs) => [
      ...msgs,
      {
        sender: "AI",
        text: "I need a few details before I can answer. Please respond to the questions below.",
      },
    ]);
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", devUserId || sessionId);

    try {
      message.loading({ content: "Processing uploaded file...", key: "upload" });
      setUploading(true);
      const response = await fetch(`${API_BASE_URL}/upload_case`, {
        method: "POST",
        body: formData,
        headers: requestHeaders,
      });
      const data = await response.json();
      setSelectedCaseId(data.case_id);
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: `✓ File uploaded successfully: ${file.name}` },
      ]);
      await loadHistory(devUserId);
      message.success({ content: `File uploaded: ${file.name}`, key: "upload" });
      return false;
    } catch (err) {
      console.error(err);
      message.error({ content: "File upload failed", key: "upload" });
      return false;
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    if (!selectedCaseId) {
      message.warning("Please upload a case file first");
      return;
    }

    setLoading(true);
    setMessages((msgs) => [...msgs, { sender: "User", text: input }]);
    const currentInput = input;
    setInput("");

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...requestHeaders },
        body: JSON.stringify({
          question: currentInput,
          case_id: selectedCaseId,
          session_id: devUserId || sessionId,
          topic: selectedTopic === "auto" ? null : selectedTopic,
        }),
      });
      const data = await response.json();

      if (data.clarification_needed) {
        startClarification(data.questions, data.missing_fields);
      } else {
        setMessages((msgs) => [
          ...msgs,
          {
            sender: "AI",
            text: data.answer,
            citations: data.citations,
          },
        ]);
      }
      await loadHistory(devUserId);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: "Error: Could not reach the server." },
      ]);
      console.error(err);
      message.error("Failed to get response");
    } finally {
      setLoading(false);
    }
  };

  const submitClarifications = async () => {
    if (loading || pendingQuestions.length === 0) return;
    setLoading(true);

    const userMsgs = clarificationAnswers.map((answer, idx) => ({
      sender: "User",
      text: `Clarification ${idx + 1}: ${answer}`,
    }));
    setMessages((msgs) => [...msgs, ...userMsgs]);

    const fields = pendingMissingFields.length ? pendingMissingFields : pendingQuestions;
    const answerMap = Object.fromEntries(
      fields.map((field, idx) => [field, clarificationAnswers[idx] || ""])
    );

    try {
      const response = await fetch(`${API_BASE_URL}/clarify`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...requestHeaders },
        body: JSON.stringify({
          answers: answerMap,
          missing_fields: pendingMissingFields,
          case_id: selectedCaseId,
          session_id: devUserId || sessionId,
        }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: data.answer, citations: data.citations },
      ]);
      setPendingQuestions([]);
      setPendingMissingFields([]);
      setClarificationAnswers([]);
      await loadHistory(devUserId);
      message.success("Clarifications submitted successfully");
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: "Error: Could not submit clarifications." },
      ]);
      console.error(err);
      message.error("Failed to submit clarifications");
    } finally {
      setLoading(false);
    }
  };

  const renderAIText = (text) => {
    if (!text) return null;

    const lines = text.split("\n");
    const sections = [];
    let currentSection = { title: null, lines: [] };

    lines.forEach((line) => {
      if (line.startsWith("## ")) {
        if (currentSection.lines.length > 0) {
          sections.push({ ...currentSection });
        }
        currentSection = { title: line.replace("## ", "").trim(), lines: [] };
      } else {
        currentSection.lines.push(line);
      }
    });

    if (currentSection.lines.length > 0) {
      sections.push(currentSection);
    }

    return (
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        {sections.map((section, idx) => {
          const content = section.lines.join("\n");
          const formattedLines = content.split("\n").map((line, lineIdx) => {
            if (!line.trim()) return null;

            const boldRegex = /\*\*(.*?)\*\*/g;
            const parts = [];
            let lastIndex = 0;
            let match;

            while ((match = boldRegex.exec(line)) !== null) {
              if (match.index > lastIndex) {
                parts.push(line.substring(lastIndex, match.index));
              }
              parts.push(<strong key={`bold-${lineIdx}-${match.index}`}>{match[1]}</strong>);
              lastIndex = match.index + match[0].length;
            }

            if (lastIndex < line.length) {
              parts.push(line.substring(lastIndex));
            }

            const finalContent = parts.length > 0 ? parts : line;

            if (line.trim().startsWith("- ") || line.trim().startsWith("• ")) {
              return (
                <div key={lineIdx} style={{ marginLeft: 16, marginBottom: 4 }}>
                  • {finalContent}
                </div>
              );
            }

            return (
              <div key={lineIdx} style={{ marginBottom: 6 }}>
                {finalContent}
              </div>
            );
          });

          return (
            <Card
              key={idx}
              size="small"
              title={section.title}
              headStyle={{ backgroundColor: "#f0f5ff", fontWeight: 600 }}
            >
              {formattedLines}
            </Card>
          );
        })}
      </Space>
    );
  };

  const renderCitations = (citations) => {
    if (!citations || citations.length === 0) return null;

    return (
      <Collapse
        size="small"
        style={{ marginTop: 12 }}
        items={[
          {
            key: "1",
            label: `📚 Referenced Cases (${citations.length})`,
            children: (
              <List
                size="small"
                dataSource={citations}
                renderItem={(cit) => (
                  <List.Item>
                    <Space direction="vertical" size="small" style={{ width: "100%" }}>
                      <Text strong>{cit.file_name}</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        Relevance: {(cit.score * 100).toFixed(1)}%
                      </Text>
                      <Paragraph
                        ellipsis={{ rows: 2, expandable: true, symbol: "more" }}
                        style={{ margin: 0, fontSize: 12 }}
                      >
                        {cit.content}
                      </Paragraph>
                    </Space>
                  </List.Item>
                )}
              />
            ),
          },
        ]}
      />
    );
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f5f7fb",
        color: "#111827",
        display: "flex",
        padding: 20,
        gap: 20,
        boxSizing: "border-box",
      }}
    >
      <aside
        style={{
          width: 320,
          minWidth: 280,
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: 16,
          padding: 16,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          boxShadow: "0 10px 30px rgba(17,24,39,0.08)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Avatar icon={<FileTextOutlined />} style={{ backgroundColor: "#2563eb" }} />
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#111827" }}>
              Law AI Assistant
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>Family law research</div>
          </div>
        </div>

        <div style={{ background: "#f9fafb", borderRadius: 12, padding: 12 }}>
          <Text strong style={{ color: "#111827" }}>User ID</Text>
          <Input
            id="dev-user-id"
            name="devUserId"
            autoComplete="off"
            placeholder="Enter user ID"
            value={devUserId}
            onChange={(e) => setDevUserId(e.target.value)}
            prefix={<UserOutlined />}
            style={{ marginTop: 8 }}
          />
        </div>

        <div style={{ background: "#f9fafb", borderRadius: 12, padding: 12 }}>
          <Upload
            beforeUpload={uploadFile}
            showUploadList={false}
            accept=".txt"
            disabled={uploading}
          >
            <Button
              icon={<UploadOutlined />}
              type="primary"
              block
              loading={uploading}
            >
              Upload Case File
            </Button>
          </Upload>
          <div style={{ fontSize: 12, color: "#6b7280", marginTop: 8 }}>
            Only .txt files supported.
          </div>
        </div>

        <div
          style={{
            background: "#f9fafb",
            borderRadius: 12,
            padding: 12,
            flex: 1,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <Text strong style={{ color: "#111827" }}>Case History</Text>
          <div style={{ marginTop: 8, overflowY: "auto" }}>
            {history.length === 0 ? (
              <div style={{ fontSize: 12, color: "#6b7280" }}>No cases found.</div>
            ) : (
              <List
                size="small"
                dataSource={history}
                renderItem={(caseItem) => (
                  <List.Item
                    style={{ cursor: "pointer", padding: "8px 0" }}
                    onClick={() => loadCaseIntoChat(caseItem)}
                  >
                    <Space direction="vertical" size={0} style={{ width: "100%" }}>
                      <Text strong ellipsis style={{ color: "#111827" }}>
                        {caseItem.filename}
                      </Text>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        {caseItem.qa?.length || 0} Q&A
                      </Text>
                    </Space>
                  </List.Item>
                )}
              />
            )}
          </div>
        </div>
      </aside>
      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          minWidth: 0,
        }}
      >
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: 18,
            padding: 16,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            boxShadow: "0 10px 30px rgba(17,24,39,0.08)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Avatar icon={<RobotOutlined />} style={{ backgroundColor: "#22c55e" }} />
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#111827" }}>
                Chat Workspace
              </div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Upload a case, ask a question, get cited answers.
              </div>
            </div>
          </div>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={resetChat}>
              Reset
            </Button>
          </Space>
        </div>

        <div
          style={{
            flex: 1,
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: 18,
            padding: 16,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            boxShadow: "0 10px 30px rgba(17,24,39,0.08)",
          }}
        >
          <div style={{ flex: 1, overflowY: "auto", paddingRight: 8 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "flex",
                    justifyContent: msg.sender === "User" ? "flex-end" : "flex-start",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      flexDirection: msg.sender === "User" ? "row-reverse" : "row",
                      alignItems: "flex-start",
                      gap: 10,
                      maxWidth: "85%",
                    }}
                  >
                    <Avatar
                      icon={msg.sender === "User" ? <UserOutlined /> : <RobotOutlined />}
                      style={{
                        backgroundColor: msg.sender === "User" ? "#3b82f6" : "#22c55e",
                      }}
                    />
                    <div
                      style={{
                        background: msg.sender === "User" ? "#e0edff" : "#ffffff",
                        border: "1px solid #e5e7eb",
                        borderRadius: 14,
                        padding: 12,
                        color: "#111827",
                        width: "100%",
                      }}
                    >
                      {msg.sender === "AI" ? renderAIText(msg.text) : <Text>{msg.text}</Text>}
                      {msg.citations && renderCitations(msg.citations)}
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div style={{ textAlign: "center" }}>
                  <Spin tip="Thinking..." />
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>

        {pendingQuestions.length > 0 && (
          <div
            style={{
              background: "#ffffff",
              border: "1px solid #e5e7eb",
              borderRadius: 16,
              padding: 16,
              boxShadow: "0 10px 30px rgba(17,24,39,0.08)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <QuestionCircleOutlined />
              <Text strong style={{ color: "#111827" }}>Clarification Required</Text>
            </div>
            <Space direction="vertical" style={{ width: "100%", marginTop: 12 }} size="middle">
              {pendingQuestions.map((q, idx) => (
                <div key={idx}>
                  <Text strong style={{ color: "#111827" }}>{q}</Text>
                  <TextArea
                    id={`clarification-${idx}`}
                    name={`clarification-${idx}`}
                    autoComplete="off"
                    rows={2}
                    placeholder="Your answer..."
                    value={clarificationAnswers[idx]}
                    onChange={(e) => {
                      const updated = [...clarificationAnswers];
                      updated[idx] = e.target.value;
                      setClarificationAnswers(updated);
                    }}
                    style={{ marginTop: 8 }}
                  />
                </div>
              ))}
              <Button type="primary" onClick={submitClarifications} loading={loading} block>
                Submit Answers
              </Button>
            </Space>
          </div>
        )}

        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: 18,
            padding: 16,
            boxShadow: "0 10px 30px rgba(17,24,39,0.08)",
          }}
        >
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
            <TextArea
              id="case-question-input"
              name="caseQuestion"
              autoComplete="off"
              autoSize={{ minRows: 3, maxRows: 8 }}
              placeholder="Ask a question about the case..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPressEnter={(e) => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              disabled={loading || !selectedCaseId}
              style={{ flex: 1, minWidth: 0 }}
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={sendMessage}
              loading={loading}
              disabled={!selectedCaseId}
              style={{ height: 44, paddingInline: 18 }}
            >
              Send
            </Button>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginTop: 10,
              flexWrap: "wrap",
            }}
          >
            <Text strong style={{ color: "#111827" }}>Topic Focus</Text>
            <Radio.Group
              value={selectedTopic}
              onChange={(e) => setSelectedTopic(e.target.value)}
            >
              <Space size="middle" wrap>
                <Radio value="auto">Auto</Radio>
                <Radio value="property_division">Property division</Radio>
                <Radio value="children_parenting">Children parenting</Radio>
                <Radio value="spousal_maintenance">Spousal maintenance</Radio>
                <Radio value="family_violence_safety">Family violence & safety</Radio>
                <Radio value="prenup_postnup">Pre/post-nuptial</Radio>
              </Space>
            </Radio.Group>
          </div>
          <div style={{ textAlign: "right", marginTop: 8 }}>
            {!selectedCaseId && (
              <Text type="secondary" style={{ marginRight: 12 }}>
                Upload a case file to enable questions.
              </Text>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default ChatWindowAntd;
