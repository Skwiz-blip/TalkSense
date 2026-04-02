export declare class ConversationEntity {
    id: string;
    organizationId: string;
    text: string;
    language: string;
    type: 'meeting' | 'email' | 'chat' | 'call';
    title: string;
    participants: string[];
    participantEmails: string[];
    modelVersion: string;
    conversationDate: Date;
    createdAt: Date;
    updatedAt: Date;
}
