import React, { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function ChatWindow() {
  const [messages, setMessages] = useState([
    { sender: "AI", text: "Hello! Ask me anything about the law cases." },
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

  const requestHeaders = useMemo(() => {
    if (!devUserId) return {};
    return { "X-DEV-USER": devUserId };
  }, [devUserId]);

  const resetChat = () => {
    setSelectedCaseId(null);
    setMessages([{ sender: "AI", text: "Hello! Ask me anything about the law cases." }]);
    setPendingQuestions([]);
    setPendingMissingFields([]);
    setClarificationAnswers([]);
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
    }
  };

  const loadCaseIntoChat = (caseItem) => {
    setSelectedCaseId(caseItem.case_id);
    setPendingQuestions([]);
    setPendingMissingFields([]);
    setClarificationAnswers([]);
    const caseMessages = (caseItem.qa || []).flatMap((qa) => [
      { sender: "User", text: qa.question },
      { sender: "AI", text: qa.answer },
    ]);
    setMessages(
      caseMessages.length
        ? caseMessages
        : [{ sender: "AI", text: "This case has no questions yet." }]
    );
  };

  useEffect(() => {
    loadHistory(devUserId);
    setPendingQuestions([]);
    setPendingMissingFields([]);
    setClarificationAnswers([]);
  }, [devUserId]);

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
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", devUserId || sessionId);

    try {
      setUploading(true);
      const response = await fetch(`${API_BASE_URL}/upload_case`, {
        method: "POST",
        body: formData,
        headers: requestHeaders,
      });
      const data = await response.json();
      setSelectedCaseId(data.case_id);  // Set the active case_id
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: `File uploaded successfully: ${file.name}` },
      ]);
      await loadHistory(devUserId);
      return data;
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: "Error uploading file." },
      ]);
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading || pendingQuestions.length > 0) return;
    
    if (!selectedCaseId) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: "Please upload a case file first." },
      ]);
      return;
    }

    setMessages((msgs) => [...msgs, { sender: "User", text: input }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...requestHeaders },
        body: JSON.stringify({
          question: input,
          case_id: selectedCaseId,
          session_id: devUserId || sessionId,  // Still send for user identification
          topic: selectedTopic === "auto" ? null : selectedTopic,
        }),
      });
      const data = await response.json();

      if (data.clarification_needed) {
        startClarification(data.questions || [], data.missing_fields || []);
        return;
      }

      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: data.answer, citations: data.citations },
      ]);
      await loadHistory(devUserId);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: "Error: Could not reach backend." },
      ]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  const updateClarificationAnswer = (index, value) => {
    setClarificationAnswers((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
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
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "AI", text: "Error: Could not submit clarifications." },
      ]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const renderAIText = (text) => {
    // Split by markdown headers to create sections
    const sections = [];
    const lines = text.split(/\r?\n/);
    let currentSection = { title: null, content: [] };

    lines.forEach((line) => {
      const headerMatch = line.match(/^##\s+(.+)$/);
      if (headerMatch) {
        // Save previous section if it has content
        if (currentSection.content.length > 0) {
          sections.push(currentSection);
        }
        // Start new section
        currentSection = { title: headerMatch[1].trim(), content: [] };
      } else if (line.trim()) {
        currentSection.content.push(line.trim());
      }
    });

    // Add the last section
    if (currentSection.content.length > 0) {
      sections.push(currentSection);
    }

    return (
      <div style={{ lineHeight: 1.6 }}>
        {sections.map((section, sectionIdx) => (
          <div key={sectionIdx} style={{ marginBottom: 20 }}>
            {section.title && (
              <div style={{
                fontSize: "16px",
                fontWeight: "bold",
                color: "#1976d2",
                marginBottom: 10,
                paddingBottom: 5,
                borderBottom: "2px solid #e0e0e0"
              }}>
                {section.title}
              </div>
            )}
            <div>
              {section.content.map((line, idx) => {
                // Bullet points
                if (line.startsWith("- ") || line.startsWith("• ")) {
                  return (
                    <div key={idx} style={{ marginBottom: 6, paddingLeft: 10 }}>
                      <span style={{ marginRight: 6 }}>•</span>
                      {line.substring(2)}
                    </div>
                  );
                }

                // Bold text
                const boldMatch = line.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
                if (boldMatch) {
                  return (
                    <div key={idx} style={{ marginBottom: 6 }}>
                      <strong>{boldMatch[1]}:</strong> {boldMatch[2]}
                    </div>
                  );
                }

                // Regular text
                return (
                  <div key={idx} style={{ marginBottom: 6 }}>
                    {line}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "16px",
        maxWidth: 1100,
        margin: "20px auto",
        fontFamily: "Arial",
      }}
    >
      <div
        style={{
          width: 280,
          border: "1px solid #ddd",
          borderRadius: "8px",
          padding: "10px",
          height: "80vh",
          overflowY: "auto",
        }}
      >
        <div style={{ marginBottom: "10px" }}>
          <label style={{ display: "block", fontSize: 12, marginBottom: 6 }}>
            Dev User ID
          </label>
          <input
            type="text"
            value={devUserId}
            onChange={(e) => setDevUserId(e.target.value)}
            placeholder="e.g. alice"
            style={{ width: "100%", padding: "8px", borderRadius: "6px" }}
          />
        </div>

        <button
          onClick={async () => {
            try {
              await fetch(`${API_BASE_URL}/reset`, {
                method: "POST",
                headers: { "Content-Type": "application/json", ...requestHeaders },
                body: JSON.stringify({ session_id: devUserId || sessionId }),
              });
            } catch (err) {
              console.error(err);
            } finally {
              resetChat();
            }
          }}
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "6px",
            border: "1px solid #ddd",
            backgroundColor: "#f5f5f5",
            marginBottom: "12px",
          }}
        >
          New chat
        </button>

        <div style={{ fontWeight: "bold", marginBottom: 8 }}>History</div>
        {history.length === 0 && (
          <div style={{ fontSize: 12, color: "#666" }}>No history yet.</div>
        )}
        {history.map((item) => (
          <button
            key={item.case_id}
            onClick={() => loadCaseIntoChat(item)}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              padding: "8px",
              borderRadius: "6px",
              border: "1px solid #eee",
              backgroundColor: selectedCaseId === item.case_id ? "#e8f5e9" : "#fff",
              marginBottom: "8px",
              cursor: "pointer",
            }}
          >
            <div style={{ fontWeight: 600 }}>
              {item.filename || "Untitled case"}
            </div>
            <div style={{ fontSize: 12, color: "#666" }}>
              {item.created_at ? new Date(item.created_at).toLocaleString() : "Unknown date"}
            </div>
          </button>
        ))}
      </div>

      <div style={{ flex: 1 }}>
        <div
          style={{
            border: "1px solid #ccc",
            padding: "10px",
            minHeight: "300px",
            borderRadius: "8px",
            marginBottom: "10px",
            overflowY: "auto",
          }}
        >
          {messages.map((msg, index) => (
            <div
              key={index}
              style={{
                textAlign: msg.sender === "AI" ? "left" : "right",
                margin: "5px 0",
              }}
            >
              <strong>{msg.sender}:</strong>{" "}
              {msg.sender === "AI" ? renderAIText(msg.text) : msg.text}

              {msg.sender === "AI" && msg.citations && (
                <div style={{ marginTop: "10px", paddingLeft: "10px" }}>
                  <em>Citations:</em>
                  <ul>
                    {msg.citations.map((citation, idx) => (
                      <li key={idx}>
                        <strong>{citation.type}:</strong> {citation.source}
                        {citation.url && (
                          <span> (<a href={citation.url} target="_blank" rel="noopener noreferrer">View</a>)</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {msg.sender === "AI" && msg.retrieved_nodes && (
                <div style={{ marginTop: "5px", paddingLeft: "10px" }}>
                  <em>Retrieved nodes:</em>
                  <ul>
                    {msg.retrieved_nodes.map((node, idx) => (
                      <li key={idx}>
                        <strong>{node.file_name}</strong> (score: {node.score.toFixed(2)}):{" "}
                        {node.content.length > 100
                          ? node.content.slice(0, 100) + "..."
                          : node.content}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          {loading && <em>AI is thinking...</em>}
        </div>

        <input
          type="file"
          onChange={(e) => uploadFile(e.target.files[0])}
          style={{ marginBottom: "10px" }}
          disabled={uploading}
        />
        {uploading && (
          <div style={{ marginBottom: "10px", fontStyle: "italic" }}>
            Processing the file...
          </div>
        )}

        <div style={{ marginBottom: "10px" }}>
          <label style={{ display: "block", fontSize: 12, marginBottom: 6 }}>
            Topic
          </label>
          <select
            value={selectedTopic}
            onChange={(e) => setSelectedTopic(e.target.value)}
            style={{ width: "100%", padding: "8px", borderRadius: "6px" }}
            disabled={pendingQuestions.length > 0}
          >
            <option value="auto">Auto detect</option>
            <option value="property_division">Property division</option>
            <option value="children_parenting">Children custody and parenting</option>
            <option value="spousal_maintenance">Spousal maintenance</option>
            <option value="family_violence_safety">Family violence and safety</option>
            <option value="prenup_postnup">Pre/post-nuptial agreement</option>
            <option value="other">Other / not listed</option>
          </select>
        </div>

        {pendingQuestions.length > 0 && (
          <div style={{ marginBottom: "12px" }}>
            <div style={{ fontWeight: "bold", marginBottom: 6 }}>
              Please answer these questions:
            </div>
            {pendingQuestions.map((q, idx) => (
              <div key={idx} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 13, marginBottom: 4 }}>{q}</div>
                <input
                  type="text"
                  value={clarificationAnswers[idx] || ""}
                  onChange={(e) => updateClarificationAnswer(idx, e.target.value)}
                  style={{ width: "100%", padding: "8px", borderRadius: "5px" }}
                />
              </div>
            ))}
            <button
              onClick={submitClarifications}
              style={{
                width: "100%",
                padding: "10px",
                borderRadius: "5px",
                backgroundColor: "#1976d2",
                color: "white",
                border: "none",
              }}
            >
              Submit clarifications
            </button>
          </div>
        )}

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question..."
          disabled={pendingQuestions.length > 0}
          style={{ width: "70%", padding: "10px", borderRadius: "5px" }}
        />
        <button
          onClick={sendMessage}
          disabled={pendingQuestions.length > 0}
          style={{
            width: "28%",
            padding: "10px",
            marginLeft: "2%",
            borderRadius: "5px",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;
