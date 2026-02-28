import gradio as gr
from dotenv import load_dotenv

from RAG_AMEX_CHAT_OPENAI.answer import answer_question

load_dotenv(override=True)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Root ────────────────────────────────────────────────── */
:root {
    --bg:           #0a0a0f;
    --bg-panel:     #111118;
    --bg-card:      #16161f;
    --bg-hover:     #1c1c28;
    --border:       rgba(255,255,255,0.07);
    --border-glow:  rgba(99,179,237,0.25);
    --gold:         #c9a84c;
    --gold-light:   #e8c96e;
    --gold-dim:     rgba(201,168,76,0.15);
    --blue:         #63b3ed;
    --blue-dim:     rgba(99,179,237,0.12);
    --text:         #e8e8f0;
    --text-muted:   #7a7a99;
    --text-dim:     #4a4a66;
    --radius:       12px;
    --radius-lg:    18px;
    --font-head:    'Syne', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --font-body:    'DM Sans', sans-serif;
    --glow-gold:    0 0 40px rgba(201,168,76,0.18);
    --glow-blue:    0 0 40px rgba(99,179,237,0.15);
    --shadow:       0 8px 32px rgba(0,0,0,0.5);
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

/* ── Animated background grid ───────────────────────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
    animation: gridDrift 20s linear infinite;
}
@keyframes gridDrift {
    0%   { background-position: 0 0; }
    100% { background-position: 48px 48px; }
}

/* ── Header ──────────────────────────────────────────────────────── */
#header-block {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 40px 24px 28px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(201,168,76,0.05) 0%, transparent 100%);
    margin-bottom: 0 !important;
}

#header-block .header-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 10px;
    display: block;
    opacity: 0.8;
}

#header-block h1 {
    font-family: var(--font-head) !important;
    font-size: clamp(1.8rem, 4vw, 2.8rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #e8e8f0 30%, var(--gold-light) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 8px !important;
    line-height: 1.1 !important;
}

#header-block p {
    font-size: 14px;
    color: var(--text-muted);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* Status badge */
#header-block .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--gold-dim);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--gold-light);
    margin-top: 14px;
    letter-spacing: 0.05em;
}
#header-block .status-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Main layout ─────────────────────────────────────────────────── */
#main-row {
    gap: 0 !important;
    position: relative;
    z-index: 1;
}

/* ── Panel wrappers ──────────────────────────────────────────────── */
#chat-panel, #context-panel {
    padding: 20px !important;
    background: var(--bg-panel) !important;
    position: relative;
}
#chat-panel {
    border-right: 1px solid var(--border);
}

/* Panel headers */
.panel-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.panel-label .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.panel-label .dot-gold  { background: var(--gold); box-shadow: 0 0 8px var(--gold); }
.panel-label .dot-blue  { background: var(--blue); box-shadow: 0 0 8px var(--blue); }

/* ── Chatbot ─────────────────────────────────────────────────────── */
#chatbot-component {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-card) !important;
    overflow: hidden !important;
}
#chatbot-component .wrap { background: transparent !important; }

/* Message bubbles */
#chatbot-component .message {
    font-family: var(--font-body) !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important;
    max-width: 88% !important;
}

/* User bubble */
#chatbot-component .message.user {
    background: linear-gradient(135deg, #1e2a3a 0%, #16233a 100%) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    box-shadow: 0 2px 12px rgba(99,179,237,0.1) !important;
    color: #d4eaff !important;
    margin-left: auto !important;
}

/* Assistant bubble */
#chatbot-component .message.bot,
#chatbot-component .message.assistant {
    background: linear-gradient(135deg, #1a1a26 0%, #141420 100%) !important;
    border: 1px solid rgba(201,168,76,0.15) !important;
    box-shadow: 0 2px 12px rgba(201,168,76,0.08) !important;
    color: var(--text) !important;
    margin-right: auto !important;
}

/* Avatar / role icons */
#chatbot-component .avatar-container img,
#chatbot-component .avatar-container svg {
    border-radius: 50% !important;
    width: 30px !important;
    height: 30px !important;
}

/* ── Input area ──────────────────────────────────────────────────── */
#input-row {
    margin-top: 12px !important;
    gap: 10px !important;
    align-items: flex-end !important;
}

#question-input textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    resize: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    min-height: 52px !important;
}
#question-input textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim), var(--glow-gold) !important;
    outline: none !important;
}
#question-input textarea::placeholder {
    color: var(--text-dim) !important;
    font-style: italic;
}
#question-input label { display: none !important; }

/* Send button */
#send-btn {
    background: linear-gradient(135deg, var(--gold) 0%, #a07830 100%) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #0a0a0f !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    padding: 0 24px !important;
    height: 52px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
    box-shadow: 0 4px 16px rgba(201,168,76,0.3) !important;
    min-width: 80px !important;
}
#send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(201,168,76,0.45) !important;
    filter: brightness(1.1) !important;
}
#send-btn:active { transform: translateY(0) !important; }

/* ── Context panel ───────────────────────────────────────────────── */
#context-panel {
    background: var(--bg-panel) !important;
}

#context-display {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
    overflow-y: auto !important;
    font-family: var(--font-body) !important;
    font-size: 13.5px !important;
    line-height: 1.7 !important;
    color: var(--text-muted) !important;
}

/* Context section titles */
#context-display h2 {
    font-family: var(--font-head) !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--gold-light) !important;
    margin: 0 0 16px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
#context-display h2::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 16px;
    background: var(--gold);
    border-radius: 2px;
    box-shadow: 0 0 8px var(--gold);
}

/* Source tags */
#context-display .source-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--blue-dim);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--blue);
    margin: 6px 0 8px;
    letter-spacing: 0.04em;
}

/* Context chunk cards */
.ctx-chunk {
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold-dim);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 12px 14px;
    margin-bottom: 14px;
    font-size: 13px;
    color: #9494b8;
    line-height: 1.7;
    transition: border-left-color 0.2s;
}
.ctx-chunk:hover { border-left-color: var(--gold); }

/* ── Suggestion chips ────────────────────────────────────────────── */
#suggestions-row {
    padding: 0 20px 16px !important;
    position: relative;
    z-index: 1;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel) !important;
}

.chip-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
}

.suggestion-chip {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12.5px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.18s;
    margin: 3px 4px 3px 0;
    font-family: var(--font-body);
    user-select: none;
}
.suggestion-chip:hover {
    border-color: var(--gold);
    color: var(--gold-light);
    background: var(--gold-dim);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(201,168,76,0.15);
}

/* ── Stats bar at bottom ─────────────────────────────────────────── */
#stats-bar {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 32px;
    padding: 14px 24px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel) !important;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 0.08em;
}
#stats-bar span { display: flex; align-items: center; gap: 6px; }
#stats-bar .val { color: var(--text-muted); }

/* ── Scrollbars ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Misc Gradio overrides ───────────────────────────────────────── */
.gr-padded { padding: 0 !important; }
footer { display: none !important; }
.gradio-container > .main { padding: 0 !important; }
"""

# ── Context formatter ──────────────────────────────────────────────────────────
def format_context(context):
    if not context:
        return """
        <div style='text-align:center; padding: 60px 20px; color: #3a3a55;'>
            <div style='font-size: 32px; margin-bottom: 12px; opacity:0.4;'>◈</div>
            <div style='font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase;'>
                Context will appear here
            </div>
            <div style='font-size: 12px; margin-top: 8px; opacity: 0.6;'>
                Ask a question to retrieve relevant documents
            </div>
        </div>
        """

    html = "<h2>Retrieved Context</h2>\n"
    for i, doc in enumerate(context):
        src = doc.metadata.get('source', 'Unknown source')
        # Shorten path if too long
        short_src = src.split("/")[-1] if "/" in src else src
        html += f"""
        <span class='source-tag'>📄 {short_src}</span>
        <div class='ctx-chunk'>{doc.page_content.strip()}</div>
        """
    return html


# ── Chat logic ─────────────────────────────────────────────────────────────────
def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


SUGGESTIONS = [
    "What products does Amex offer?",
    "Who is the CEO?",
    "Tell me about Membership Rewards",
    "Explain the Platinum Card benefits",
    "What are Amex's financials?",
    "How does fraud detection work?",
]

# ── UI build ───────────────────────────────────────────────────────────────────
def main():
    def put_message_in_chatbot(message, history):
        if not message.strip():
            return "", history
        return "", history + [{"role": "user", "content": message}]

    def use_suggestion(suggestion_text, history):
        return suggestion_text, history + [{"role": "user", "content": suggestion_text}]

    theme = gr.themes.Base(
        font=["DM Sans", "sans-serif"],
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0a0a0f",
        body_text_color="#e8e8f0",
        block_background_fill="#16161f",
        block_border_width="1px",
        block_border_color="rgba(255,255,255,0.07)",
        input_background_fill="#16161f",
        input_border_color="rgba(255,255,255,0.07)",
    )

    with gr.Blocks(title="Amex Knowledge Assistant", theme=theme, css=CSS) as ui:

        # ── Header ──────────────────────────────────────────────────
        with gr.Row(elem_id="header-block"):
            gr.HTML("""
                <span class="header-eyebrow">Powered by RAG · Vector Search · LLM</span>
                <h1>American Express<br>Knowledge Assistant</h1>
                <p>Ask anything across company, products, financials, contracts & more</p>
                <span class="status-badge">KNOWLEDGE BASE ONLINE · 22 DOCUMENTS</span>
            """)

        # ── Suggestion chips ─────────────────────────────────────────
        with gr.Row(elem_id="suggestions-row"):
            with gr.Column():
                gr.HTML('<div class="chip-label">Try asking</div>')
                chips_html = "".join(
                    f'<span class="suggestion-chip" onclick="fillQuestion(this)">{s}</span>'
                    for s in SUGGESTIONS
                )
                gr.HTML(f'<div id="chips-container">{chips_html}</div>')

        # ── Main two-column layout ───────────────────────────────────
        with gr.Row(elem_id="main-row", equal_height=True):

            # Left — Chat
            with gr.Column(scale=1, elem_id="chat-panel"):
                gr.HTML('<div class="panel-label"><span class="dot dot-gold"></span>Conversation</div>')

                chatbot = gr.Chatbot(
                    elem_id="chatbot-component",
                    label="",
                    height=520,
                    type="messages",
                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=(
                        None,  # user
                        None,  # assistant — can swap for image URLs
                    ),
                    placeholder="<div style='text-align:center;color:#3a3a55;padding:40px;font-family:DM Mono,monospace;font-size:12px;letter-spacing:0.1em'>NO MESSAGES YET<br><br><span style='opacity:0.5;font-size:11px'>Ask your first question below</span></div>",
                )

                with gr.Row(elem_id="input-row"):
                    message = gr.Textbox(
                        elem_id="question-input",
                        placeholder="Ask anything about American Express…",
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        scale=5,
                    )
                    send_btn = gr.Button(
                        "Send ↵",
                        elem_id="send-btn",
                        scale=1,
                        variant="primary",
                    )

            # Right — Context
            with gr.Column(scale=1, elem_id="context-panel"):
                gr.HTML('<div class="panel-label"><span class="dot dot-blue"></span>Retrieved Context</div>')

                context_display = gr.HTML(
                    elem_id="context-display",
                    value="""
                    <div style='text-align:center;padding:60px 20px;color:#3a3a55;'>
                        <div style='font-size:32px;margin-bottom:12px;opacity:0.4;'>◈</div>
                        <div style='font-family:DM Mono,monospace;font-size:12px;letter-spacing:0.1em;text-transform:uppercase;'>
                            Context will appear here
                        </div>
                        <div style='font-size:12px;margin-top:8px;opacity:0.6;'>
                            Ask a question to retrieve relevant documents
                        </div>
                    </div>
                    """,
                )

        # ── Stats bar ────────────────────────────────────────────────
        with gr.Row(elem_id="stats-bar"):
            gr.HTML("""
                <span>DOCUMENTS <span class="val">22</span></span>
                <span>CATEGORIES <span class="val">8</span></span>
                <span>MODEL <span class="val">RAG · VECTOR</span></span>
                <span>ENTITY <span class="val">NYSE: AXP</span></span>
            """)

        # ── JS: chip click fills textbox ─────────────────────────────
        gr.HTML("""
        <script>
        function fillQuestion(el) {
            const ta = document.querySelector('#question-input textarea');
            if (!ta) return;
            ta.value = el.innerText;
            ta.dispatchEvent(new Event('input', {bubbles:true}));
            ta.focus();
            // highlight the chip briefly
            el.style.borderColor = '#c9a84c';
            el.style.color = '#e8c96e';
            setTimeout(() => {
                el.style.borderColor = '';
                el.style.color = '';
            }, 600);
        }
        // allow Enter to submit (Shift+Enter for newline)
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                const ta = document.activeElement;
                if (ta && ta.id && ta.closest('#question-input')) {
                    e.preventDefault();
                    document.querySelector('#send-btn')?.click();
                }
            }
        });
        </script>
        """)

        # ── Event wiring ─────────────────────────────────────────────
        submit_args = dict(
            fn=put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        )
        respond_args = dict(
            fn=chat,
            inputs=chatbot,
            outputs=[chatbot, context_display],
        )

        message.submit(**submit_args).then(**respond_args)
        send_btn.click(**submit_args).then(**respond_args)

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
