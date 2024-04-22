/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 */
package com.datastax.ai.agent.vector;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import com.datastax.ai.agent.base.AiAgent;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.CassandraVectorStore;

import reactor.core.publisher.Flux;


public class AiAgentSessionVector implements AiAgent {

    private static final Logger logger = LoggerFactory.getLogger(AiAgentSessionVector.class);

    private final AiAgent agent;
    private final CassandraVectorStore store;

    public static AiAgentSessionVector create(AiAgent agent, CassandraVectorStore store) {
        return new AiAgentSessionVector(agent, store);
    }

    AiAgentSessionVector(AiAgent agent, CassandraVectorStore store) {
        this.agent = agent;
        this.store = store;
    }

    @Override
    public Prompt createPrompt(UserMessage message, Map<String,Object> promptProperties) {
        List<Document> similarDocuments = store.similaritySearch(message.getContent());
        promptProperties = promptProperties(promptProperties);

        // re-ranking happens here

        promptProperties.put("documents", similarDocuments);
        return agent.createPrompt(message, promptProperties);
    }

    @Override
    public Flux<ChatResponse> send(Prompt prompt) {

        final AtomicReference<StringBuilder> stringBufferRef = new AtomicReference<>();
        final Document input = new Document(prompt.getContents());

        return agent.send(prompt).doOnSubscribe(subscription -> {
            stringBufferRef.set(new StringBuilder());
          }).doOnNext(chatResponse -> {
            if (null != chatResponse.getResult()) {
              if (null != chatResponse.getResult().getOutput().getContent()) {
                stringBufferRef.get().append(chatResponse.getResult().getOutput().getContent());
              }
            }
          }).doOnComplete(() -> {

            // TODO – metadata: user/session/…?
            // FIXME – how to cross-reference input and output
            Document output = new Document(stringBufferRef.get().toString());
            store.add(List.of(input, output));

            stringBufferRef.set(null);
          }).doOnError(e -> {
            logger.error("Aggregation Error", e);
            stringBufferRef.set(null);
          });
    }

    @Override
    public Map<String,Object> promptProperties(Map<String,Object> promptProperties) {
        return agent.promptProperties(promptProperties);
    }

}
