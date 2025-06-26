import React, { useState, useEffect } from 'react';

const BACKEND_URL = 'http://localhost:8001';

// Inline styles for the application
const styles = {
  app: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
  },
  header: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
    boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
    padding: '1rem 2rem',
  },
  headerContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '1.8rem',
    fontWeight: '700',
    color: '#2d3748',
    margin: 0,
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  version: {
    background: 'linear-gradient(135deg, #48bb78, #38a169)',
    color: 'white',
    padding: '0.25rem 0.75rem',
    borderRadius: '9999px',
    fontSize: '0.75rem',
    fontWeight: '600',
  },
  subtitle: {
    color: '#718096',
    fontSize: '0.9rem',
  },
  navigation: {
    background: 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
  },
  navContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 2rem',
    display: 'flex',
    gap: '2rem',
  },
  navButton: {
    padding: '1rem 0',
    background: 'none',
    border: 'none',
    borderBottom: '3px solid transparent',
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#718096',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  navButtonActive: {
    color: '#667eea',
    borderBottomColor: '#667eea',
  },
  mainContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem',
  },
  contentWrapper: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '16px',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
    overflow: 'hidden',
  },
  card: {
    padding: '2rem',
  },
  cardTitle: {
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#2d3748',
    margin: '0 0 1.5rem 0',
    borderBottom: '2px solid #e2e8f0',
    paddingBottom: '0.75rem',
  },
  statusContent: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1.5rem',
  },
  statusItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  label: {
    fontWeight: '500',
    color: '#4a5568',
    minWidth: '120px',
  },
  statusBadge: {
    padding: '0.25rem 0.75rem',
    borderRadius: '9999px',
    fontSize: '0.75rem',
    fontWeight: '600',
  },
  statusBadgeHealthy: {
    background: '#c6f6d5',
    color: '#22543d',
  },
  statusBadgeUnhealthy: {
    background: '#fed7d7',
    color: '#742a2a',
  },
  servicesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '0.75rem',
  },
  serviceItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.75rem',
    background: '#f7fafc',
    borderRadius: '8px',
    border: '1px solid #e2e8f0',
  },
  serviceStatus: {
    padding: '0.25rem 0.5rem',
    borderRadius: '6px',
    fontSize: '0.75rem',
    fontWeight: '600',
  },
  serviceStatusAvailable: {
    background: '#c6f6d5',
    color: '#22543d',
  },
  serviceStatusUnavailable: {
    background: '#fed7d7',
    color: '#742a2a',
  },
  loading: {
    textAlign: 'center' as const,
    padding: '3rem',
    color: '#718096',
  },
  spinner: {
    width: '2rem',
    height: '2rem',
    border: '3px solid #e2e8f0',
    borderTop: '3px solid #667eea',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    margin: '0 auto 1rem',
  },
  form: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1.5rem',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.5rem',
  },
  formLabel: {
    fontWeight: '500',
    color: '#4a5568',
    fontSize: '0.9rem',
  },
  formInput: {
    padding: '0.75rem',
    border: '2px solid #e2e8f0',
    borderRadius: '8px',
    fontSize: '0.9rem',
    transition: 'border-color 0.2s ease',
    background: 'white',
  },
  submitButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    fontSize: '0.9rem',
  },
  submitButtonDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  errorMessage: {
    padding: '1rem',
    background: '#fed7d7',
    border: '1px solid #feb2b2',
    borderRadius: '8px',
    color: '#742a2a',
    fontSize: '0.9rem',
  },
  resultSection: {
    marginTop: '1.5rem',
    padding: '1.5rem',
    background: '#f7fafc',
    borderRadius: '8px',
    border: '1px solid #e2e8f0',
  },
  chatContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    height: '500px',
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '1rem',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
  },
  message: {
    maxWidth: '70%',
    padding: '0.75rem 1rem',
    borderRadius: '12px',
    fontSize: '0.9rem',
    lineHeight: 1.4,
  },
  userMessage: {
    alignSelf: 'flex-end' as const,
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    color: 'white',
  },
  assistantMessage: {
    alignSelf: 'flex-start' as const,
    background: '#f7fafc',
    color: '#4a5568',
    border: '1px solid #e2e8f0',
  },
  chatInputForm: {
    display: 'flex',
    gap: '0.5rem',
    padding: '1rem',
    borderTop: '1px solid #e2e8f0',
    background: '#f7fafc',
  },
  chatInput: {
    flex: 1,
    padding: '0.75rem',
    border: '2px solid #e2e8f0',
    borderRadius: '8px',
    fontSize: '0.9rem',
  },
  sendButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontWeight: '600',
    cursor: 'pointer',
  },
};

function App() {
  const [activeTab, setActiveTab] = useState('status');
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/health`);
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const tabs = [
    { id: 'status', name: 'Status', icon: '🏦' },
    { id: 'upload', name: 'Document Upload', icon: '📄' },
    { id: 'analytics', name: 'Risk Analytics', icon: '📊' },
    { id: 'assistant', name: 'Digital Assistant', icon: '🤖' },
  ];

  return (
    <div style={styles.app}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div>
            <h1 style={styles.title}>
              <span role="img" aria-label="bank">🏦</span>
              FinDoc AI
              <span style={styles.version}>v1.0.0</span>
            </h1>
            <div style={styles.subtitle}>
              Latin American Bank Document Processing Platform
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav style={styles.navigation}>
        <div style={styles.navContent}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                ...styles.navButton,
                ...(activeTab === tab.id ? styles.navButtonActive : {})
              }}
            >
              <span>{tab.icon}</span>
              {tab.name}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main style={styles.mainContent}>
        <div style={styles.contentWrapper}>
          {activeTab === 'status' && <StatusTab status={status} />}
          {activeTab === 'upload' && <UploadTab />}
          {activeTab === 'analytics' && <AnalyticsTab />}
          {activeTab === 'assistant' && <AssistantTab />}
        </div>
      </main>
    </div>
  );
}

function StatusTab({ status }: { status: any }) {
  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>Platform Status</h3>
      
      {status ? (
        <div style={styles.statusContent}>
          <div style={styles.statusItem}>
            <span style={styles.label}>Overall Status:</span>
            <span style={{
              ...styles.statusBadge,
              ...(status.status === 'healthy' ? styles.statusBadgeHealthy : styles.statusBadgeUnhealthy)
            }}>
              {status.status}
            </span>
          </div>

          <div>
            <h4 style={{ margin: '0 0 0.75rem 0', color: '#4a5568' }}>Services:</h4>
            <div style={styles.servicesGrid}>
              {Object.entries(status.services).map(([service, available]) => (
                <div key={service} style={styles.serviceItem}>
                  <span style={{ textTransform: 'capitalize' }}>{service.replace(/_/g, ' ')}</span>
                  <span style={{
                    ...styles.serviceStatus,
                    ...(available ? styles.serviceStatusAvailable : styles.serviceStatusUnavailable)
                  }}>
                    {available ? 'Available' : 'Unavailable'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div style={styles.loading}>
          <div style={styles.spinner}></div>
          <p>Loading status...</p>
        </div>
      )}
    </div>
  );
}

function UploadTab() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/documents/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>Document Upload & Processing</h3>
      
      <form onSubmit={handleUpload} style={styles.form}>
        <div style={styles.formGroup}>
          <label style={styles.formLabel}>Select Document</label>
          <input
            type="file"
            accept=".pdf,.jpg,.jpeg,.png,.tiff"
            onChange={handleFileChange}
            style={styles.formInput}
          />
        </div>

        <button
          type="submit"
          disabled={!file || uploading}
          style={{
            ...styles.submitButton,
            ...(uploading ? styles.submitButtonDisabled : {})
          }}
        >
          {uploading ? 'Processing...' : 'Upload & Process Document'}
        </button>
      </form>

      {error && (
        <div style={styles.errorMessage}>
          {error}
        </div>
      )}

      {result && (
        <div style={styles.resultSection}>
          <h4>Processing Result:</h4>
          <pre style={{ background: '#2d3748', color: '#e2e8f0', padding: '1rem', borderRadius: '6px', overflow: 'auto' }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function AnalyticsTab() {
  const [formData, setFormData] = useState({
    document_id: '',
    customer_data: '{"name": "Juan Perez", "age": 35, "income": 50000}',
    loan_amount: '10000',
    loan_type: 'personal'
  });
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/analytics/risk-score`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          customer_data: JSON.parse(formData.customer_data),
          loan_amount: parseFloat(formData.loan_amount),
        }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>Risk Analytics</h3>
      
      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.formGroup}>
          <label style={styles.formLabel}>Document ID</label>
          <input
            type="text"
            name="document_id"
            value={formData.document_id}
            onChange={handleChange}
            style={styles.formInput}
            placeholder="Enter document ID"
          />
        </div>

        <div style={styles.formGroup}>
          <label style={styles.formLabel}>Customer Data (JSON)</label>
          <textarea
            name="customer_data"
            value={formData.customer_data}
            onChange={handleChange}
            rows={3}
            style={styles.formInput}
            placeholder="Enter customer data as JSON"
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div style={styles.formGroup}>
            <label style={styles.formLabel}>Loan Amount</label>
            <input
              type="number"
              name="loan_amount"
              value={formData.loan_amount}
              onChange={handleChange}
              style={styles.formInput}
              placeholder="Enter loan amount"
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.formLabel}>Loan Type</label>
            <select
              name="loan_type"
              value={formData.loan_type}
              onChange={handleChange}
              style={styles.formInput}
            >
              <option value="personal">Personal</option>
              <option value="business">Business</option>
              <option value="mortgage">Mortgage</option>
              <option value="auto">Auto</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={analyzing}
          style={{
            ...styles.submitButton,
            ...(analyzing ? styles.submitButtonDisabled : {})
          }}
        >
          {analyzing ? 'Analyzing...' : 'Calculate Risk Score'}
        </button>
      </form>

      {error && (
        <div style={styles.errorMessage}>
          {error}
        </div>
      )}

      {result && (
        <div style={styles.resultSection}>
          <h4>Risk Analysis Result:</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
            <div style={{ padding: '0.75rem', background: 'white', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <span style={{ fontWeight: '500', color: '#4a5568' }}>Risk Score:</span>
              <span style={{ fontWeight: '600', color: '#2d3748', marginLeft: '0.5rem' }}>{result.risk_score}</span>
            </div>
            <div style={{ padding: '0.75rem', background: 'white', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <span style={{ fontWeight: '500', color: '#4a5568' }}>Category:</span>
              <span style={{ fontWeight: '600', color: '#2d3748', marginLeft: '0.5rem' }}>{result.risk_category}</span>
            </div>
            <div style={{ padding: '0.75rem', background: 'white', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <span style={{ fontWeight: '500', color: '#4a5568' }}>Confidence:</span>
              <span style={{ fontWeight: '600', color: '#2d3748', marginLeft: '0.5rem' }}>{(result.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function AssistantTab() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Array<{type: 'user' | 'assistant', content: string}>>([]);
  const [sending, setSending] = useState(false);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage = message;
    setMessage('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setSending(true);

    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/assistant/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          language: 'es'
        }),
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, { type: 'assistant', content: data.response }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div style={styles.card}>
      <h3 style={styles.cardTitle}>Digital Assistant</h3>
      
      <div style={styles.chatContainer}>
        <div style={styles.messagesContainer}>
          {messages.length === 0 ? (
            <div style={{ textAlign: 'center', color: '#718096', padding: '2rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🤖</div>
              <p>Ask me anything about your documents or financial analysis!</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                style={{
                  ...styles.message,
                  ...(msg.type === 'user' ? styles.userMessage : styles.assistantMessage)
                }}
              >
                {msg.content}
              </div>
            ))
          )}
          
          {sending && (
            <div style={{ ...styles.message, ...styles.assistantMessage }}>
              <span style={{ fontStyle: 'italic', color: '#718096' }}>Thinking...</span>
            </div>
          )}
        </div>

        <form onSubmit={handleSend} style={styles.chatInputForm}>
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            style={styles.chatInput}
            disabled={sending}
          />
          <button
            type="submit"
            disabled={!message.trim() || sending}
            style={{
              ...styles.sendButton,
              ...(sending ? styles.submitButtonDisabled : {})
            }}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
