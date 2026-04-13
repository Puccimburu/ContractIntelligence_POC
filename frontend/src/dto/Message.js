// Mirrors the Message DTO from the original UI
export default class Message {
  constructor(data = {}) {
    this.id = data.messageId || data.id || '';
    this.role = data.role || 'user';
    this.attachments = data.attachments || [];
    this.followUpQuestions = data.followUpQuestions || [];
    this.reasoning = data.reasoning || '';
    this.citations = data.citations || [];

    // Normalize content to always be an array of {type, content} objects
    const raw = data.content;
    if (!raw || (Array.isArray(raw) && raw.length === 0)) {
      this.content = [];
    } else if (typeof raw === 'string') {
      try {
        const parsed = JSON.parse(raw);
        this.content = Array.isArray(parsed) ? parsed : [{ type: 'text', content: raw }];
      } catch (_) {
        this.content = [{ type: 'text', content: raw }];
      }
    } else if (Array.isArray(raw)) {
      this.content = raw;
    } else {
      this.content = [{ type: 'text', content: String(raw) }];
    }
  }

  getText() {
    const textItem = this.content.find(c => c.type === 'text');
    return textItem ? textItem.content : '';
  }
}
