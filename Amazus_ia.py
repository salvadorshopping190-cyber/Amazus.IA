"""
Amazus IA - Assistente Virtual Empática com Aprendizado Rápido
Requer: transformers, torch, requests, pysentimiento (opcional)
Instale com: pip install transformers torch requests pysentimiento
"""

import torch
import random
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import requests

class AmazusIA:
    def __init__(self, google_api_key=None, google_cx=None):
        # Carregar modelo GPT-2 em português (pequeno, mas ajustável)
        self.model_name = "pierreguillou/gpt2-small-portuguese"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configuração do Google Custom Search (necessário chave de API e CX)
        self.google_api_key = google_api_key
        self.google_cx = google_cx

        # Filtro de segurança (palavras-chave bloqueadas)
        self.blocklist = ["crime", "assassinato", "drogas", "roubar", "ilegal", 
                          "fraude", "violência", "suicídio", "automutilação"]
        # Resposta segura padrão
        self.safe_response = "Sinto muito, mas não posso ajudar com isso. Se você está passando por um momento difícil, posso conversar ou sugerir ajuda profissional."

        # Histórico de conversas para aprendizado
        self.conversation_log = []
        self.log_file = "conversas_amazus.json"

        # Pipeline de análise de sentimentos (opcional, para detectar tristeza)
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                               model="pysentimiento/bertweet-pt-sentiment")
        except:
            self.sentiment_analyzer = None

    def search_google(self, query):
        """Busca na internet usando a API do Google Custom Search"""
        if not self.google_api_key or not self.google_cx:
            return ""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": 1
        }
        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            if "items" in data:
                snippet = data["items"][0].get("snippet", "")
                return snippet[:500]  # limita tamanho
        except:
            pass
        return ""

    def moderate_input(self, text):
        """Verifica se a entrada contém conteúdo bloqueado"""
        text_lower = text.lower()
        for palavra in self.blocklist:
            if palavra in text_lower:
                return True
        return False

    def generate_response(self, user_input, use_search=False):
        """
        Gera uma resposta com base na entrada do usuário.
        Se necessário, realiza busca na web para enriquecer a resposta.
        """
        # Filtro de segurança
        if self.moderate_input(user_input):
            return self.safe_response

        # Contexto psicológico empático
        system_prompt = (
            "Você é a Amazus IA, uma psicóloga virtual empática, carinhosa e profissional. "
            "Você conversa com pessoas que podem estar tristes ou ansiosas, oferecendo apoio, "
            "escutando atentamente e nunca julgando. Respostas curtas e acolhedoras.\n"
        )
        
        # Se a pergunta parece buscar informação factual (contém ? ou palavras-chave), tenta buscar no Google
        context_from_web = ""
        if use_search and ("?" in user_input or any(p in user_input.lower() for p in ["pesquise", "busque", "procure"])):
            context_from_web = self.search_google(user_input)
            if context_from_web:
                system_prompt += f"Informação da internet: {context_from_web}\n"

        # Prepara o prompt para o modelo
        prompt = f"{system_prompt}Usuário: {user_input}\nAmazus IA:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Geração com parâmetros adequados
        outputs = self.model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrai apenas a parte da resposta (após "Amazus IA:")
        if "Amazus IA:" in response:
            response = response.split("Amazus IA:")[-1].strip()
        else:
            response = response[len(prompt):].strip()

        # Filtro final na resposta gerada (evita ajudar em algo errado)
        if self.moderate_input(response):
            return self.safe_response

        return response

    def learn_from_interaction(self, user_input, ia_response):
        """
        Aprendizado rápido: armazena a conversa e periodicamente re-treina o modelo.
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "amazus": ia_response
        }
        self.conversation_log.append(interaction)
        # Salva em arquivo para treinamento posterior
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(interaction, ensure_ascii=False) + "\n")

    def train_on_recent_conversations(self):
        """
        Ajuste fino incremental usando as conversas registradas.
        (Simula aprendizado rápido - ideal rodar a cada N interações)
        """
        if len(self.conversation_log) < 10:
            return  # precisa de um mínimo de exemplos

        # Cria dataset de treino: cada linha é uma interação formatada
        train_lines = []
        for conv in self.conversation_log[-100:]:  # usa as últimas 100 interações
            line = f"Usuário: {conv['user']}\nAmazus IA: {conv['amazus']}"
            train_lines.append(line)

        train_file = "temp_train_amazus.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_lines))

        # Configura dataset e treinamento rápido
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=128
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir="./amazus_checkpoint",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10,
            save_total_limit=1,
            logging_steps=5,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        trainer.train()
        # Atualiza o modelo com os novos pesos
        self.model = trainer.model
        print("✔ Aprendizado concluído! Modelo atualizado com novas conversas.")

    def chat(self):
        """Loop de conversa interativo."""
        print("🧠 Amazus IA iniciada. Digite 'sair' para encerrar.")
        interaction_count = 0
        while True:
            user_input = input("\nVocê: ")
            if user_input.lower() in ["sair", "exit", "tchau"]:
                print("Amazus IA: Até logo! Se precisar conversar, estarei aqui.")
                break

            # Gera resposta com acesso à internet ativado por padrão
            resposta = self.generate_response(user_input, use_search=True)
            print(f"Amazus IA: {resposta}")

            # Aprende com a interação
            self.learn_from_interaction(user_input, resposta)
            interaction_count += 1

            # A cada 15 interações, realiza treinamento rápido (simula aprendizado contínuo)
            if interaction_count % 15 == 0 and len(self.conversation_log) >= 10:
                print("⏳ Aprendendo com as conversas recentes...")
                self.train_on_recent_conversations()

if __name__ == "__main__":
    # Para executar, configure sua chave da API do Google se quiser buscas na web
    # Exemplo: ai = AmazusIA(google_api_key="SUA_KEY", google_cx="SEU_CX")
    # Sem a chave, a IA funciona como psicóloga sem acesso à internet.
    ai = AmazusIA()
    ai.chat()
