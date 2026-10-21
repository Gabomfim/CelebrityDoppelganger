(() => {
  const STORAGE_KEY = "celebrity-twin-locale";
  const portuguese = {
    "How it works": "Como funciona",
    "AI-powered face matching": "Reconhecimento facial com IA",
    "Meet your": "Conheça seus",
    "celebrity twins.": "sósias famosos.",
    "Three selfies. One personal prototype. Your three most convincing famous lookalikes.": "Três selfies. Um protótipo pessoal. Seus três sósias famosos mais convincentes.",
    "Private by design": "Privacidade desde o projeto",
    "— your selfies are processed in memory and never stored on our server.": "— suas selfies são processadas em memória e nunca armazenadas em nosso servidor.",
    "Put your best face forward": "Mostre seu melhor ângulo",
    "Our photobooth takes three quick selfies and combines them into one more reliable face prototype.": "Nossa cabine tira três selfies rápidas e as combina em um protótipo facial mais confiável.",
    "Remove sunglasses before starting": "Retire os óculos escuros antes de começar",
    "Keep exactly one face in the frame": "Mantenha apenas um rosto no enquadramento",
    "Center your face and make a small expression change each shot": "Centralize o rosto e mude levemente a expressão a cada foto",
    "Private session": "Sessão privada",
    "Face studio": "Estúdio facial",
    "Place your face here": "Posicione seu rosto aqui",
    "Ready for your close-up?": "Pronto para o seu close?",
    "Tap “Start camera” and allow access when asked.": "Toque em “Abrir câmera” e permita o acesso quando solicitado.",
    "Start camera": "Abrir câmera",
    "Take 3 selfies": "Tirar 3 selfies",
    "Upload 3 photos": "Enviar 3 fotos",
    "Find my celebrity twins": "Encontrar meus sósias famosos",
    "🔒 All three images and the temporary user prototype are discarded immediately after matching. We don't create accounts or retain biometric data.": "🔒 As três imagens e o protótipo temporário são descartados imediatamente após a comparação. Não criamos contas nem guardamos dados biométricos.",
    "The verdict is in": "Saiu o resultado",
    "Your famous lookalikes": "Seus sósias famosos",
    "Your similarity orbit": "Sua órbita de semelhança",
    "See why these three faces are closest.": "Veja por que estes três rostos são os mais próximos.",
    "Every point lies on the unit circle. The angle from “You” preserves each match's cosine similarity: a smaller angle means a closer face.": "Todos os pontos ficam no círculo unitário. O ângulo a partir de “Você” preserva a similaridade cosseno de cada resultado: um ângulo menor indica um rosto mais próximo.",
    "The wheel preserves each celebrity's cosine distance to you; distances between celebrity points are not represented.": "A roda preserva a distância cosseno de cada celebridade até você; as distâncias entre as celebridades não são representadas.",
    "Scroll to zoom · drag to pan · hover or tap any point": "Role para ampliar · arraste para mover · passe o cursor ou toque em qualquer ponto",
    "Share my #1 reveal": "Compartilhar meu sósia nº 1",
    "Try another photobooth": "Tentar outra sessão",
    "Your red carpet reveal": "Sua revelação no tapete vermelho",
    "Make it official.": "Conte para todo mundo.",
    "We created a BuzzFeed-style reveal card using only your celebrity match. Your selfies are never included or stored.": "Criamos um card de revelação usando apenas seu sósia famoso. Suas selfies nunca são incluídas nem armazenadas.",
    "More": "Mais",
    "↓ Download image": "↓ Baixar imagem",
    "Copy caption": "Copiar legenda",
    "Made by Gabriel Silveira with contrastive supervised prototype learning.": "Feito por Gabriel Silveira com aprendizado contrastivo supervisionado por protótipos.",
    "Made by Gabriel Silveira with supervised contrastive prototype learning.": "Feito por Gabriel Silveira com aprendizado contrastivo supervisionado por protótipos.",
    "Source code ↗": "Código-fonte ↗",
    "Behind the velvet rope": "Por trás dos bastidores",
    "Teaching faces to find their": "Ensinando rostos a encontrar seu",
    "place.": "lugar.",
    "Celebrity Twin is not trained to memorize a single perfect portrait. It learns an embedding space where photos of the same person gather closely and different identities remain apart.": "O Celebrity Twin não é treinado para memorizar um retrato perfeito. Ele aprende um espaço de embeddings no qual fotos da mesma pessoa ficam próximas e identidades diferentes permanecem separadas.",
    "Try the photobooth": "Testar a cabine",
    "See the experiment ↓": "Ver o experimento ↓",
    "One identity": "Uma identidade",
    "one compact neighborhood": "uma vizinhança compacta",
    "Within-class distance": "Distância dentro da classe",
    "more compact": "mais compacto",
    "Separation ratio": "Razão de separação",
    "+79% improvement": "+79% de melhoria",
    "Validation accuracy": "Acurácia de validação",
    "+14.8 points": "+14,8 pontos",
    "Celebrity identities": "Identidades de celebridades",
    "prototype classes": "classes de protótipos",
    "01 · Representation learning": "01 · Aprendizado de representações",
    "Classification gets the answer. SupCon shapes the space.": "A classificação encontra a resposta. SupCon molda o espaço.",
    "Cross-entropy rewards the model for choosing the correct label. That can produce accurate decision boundaries without requiring samples from the same person to sit tightly together. Supervised contrastive learning optimizes the geometry we actually use at inference time.": "A entropia cruzada recompensa o modelo por escolher o rótulo correto. Isso pode produzir boas fronteiras de decisão sem exigir que amostras da mesma pessoa fiquem próximas. O aprendizado contrastivo supervisionado otimiza a geometria usada na inferência.",
    "Cross-entropy baseline": "Baseline de entropia cruzada",
    "Correct, but loosely arranged": "Correto, mas pouco agrupado",
    "The classifier learns which side of a boundary each face belongs on. Embeddings from neighboring identities can still be diffuse or close together.": "O classificador aprende de que lado da fronteira cada rosto fica. Embeddings de identidades vizinhas ainda podem estar dispersos ou próximos demais.",
    "Selected approach": "Abordagem escolhida",
    "Compact and deliberately separated": "Compacto e deliberadamente separado",
    "Each face acts as an anchor. Other augmented images of the same celebrity are positives; every different identity is a negative. The loss pulls positives closer and pushes negatives away.": "Cada rosto atua como âncora. Outras imagens aumentadas da mesma celebridade são positivas; cada identidade diferente é negativa. A perda aproxima os positivos e afasta os negativos.",
    "02 · The training recipe": "02 · A receita de treinamento",
    "A face-specialized foundation, carefully fine-tuned.": "Uma base especializada em rostos, ajustada com cuidado.",
    "The backbone starts with facial features learned from VGGFace2. Training then adapts that representation to the celebrity dataset while preserving a normalized metric space.": "O backbone começa com características faciais aprendidas no VGGFace2. O treinamento adapta essa representação ao conjunto de celebridades, preservando um espaço métrico normalizado.",
    "Detect and align": "Detectar e alinhar",
    "Create two views": "Criar duas visões",
    "Learn normalized embeddings": "Aprender embeddings normalizados",
    "Balance every batch": "Balancear cada lote",
    "Select by geometry": "Selecionar pela geometria",
    "A face detector isolates one clear face. Every crop receives the same FaceNet pixel standardization used in training and production.": "Um detector isola um rosto nítido. Cada recorte recebe a mesma padronização de pixels do FaceNet usada no treinamento e em produção.",
    "Independent crops, flips, color changes, blur, and small geometric variations teach the model to ignore pose and lighting noise.": "Recortes, espelhamentos, mudanças de cor, desfoque e pequenas variações geométricas ensinam o modelo a ignorar ruídos de pose e iluminação.",
    "An InceptionResnetV1 backbone pretrained on VGGFace2 feeds a projection head. L2 normalization places every embedding on the same unit hypersphere.": "Um backbone InceptionResnetV1 pré-treinado no VGGFace2 alimenta uma cabeça de projeção. A normalização L2 coloca cada embedding na mesma hiperesfera unitária.",
    "Each batch contains 16 identities with 4 samples per identity and two augmented views, ensuring every anchor has meaningful positives and negatives.": "Cada lote contém 16 identidades, 4 amostras por identidade e duas visões aumentadas, garantindo positivos e negativos relevantes para cada âncora.",
    "Checkpoints are chosen using the validation separation ratio—not training accuracy alone—so compact clusters and distance from neighboring classes drive selection.": "Os checkpoints são escolhidos pela razão de separação na validação — não apenas pela acurácia de treino — para que clusters compactos e distantes das classes vizinhas orientem a seleção.",
    "Supervised contrastive objective": "Objetivo contrastivo supervisionado",
    "For each anchor, the loss rewards high similarity to every same-class positive and penalizes similarity to all other faces in the batch. A temperature of": "Para cada âncora, a perda recompensa a alta similaridade com positivos da mesma classe e penaliza a similaridade com os demais rostos do lote. Uma temperatura de",
    "controls how strongly the model focuses on difficult neighbors.": "controla quanto o modelo se concentra em vizinhos difíceis.",
    "03 · Evidence, not intuition": "03 · Evidência, não intuição",
    "SupCon won on both geometry and validation accuracy.": "SupCon venceu na geometria e na acurácia de validação.",
    "The comparison used a stratified validation split and the same pretrained backbone. The winning setup ran for 20 epochs before a final production refit on all available images.": "A comparação usou uma divisão estratificada de validação e o mesmo backbone pré-treinado. A configuração vencedora rodou por 20 épocas antes do ajuste final com todas as imagens disponíveis.",
    "Validation geometry over time.": "Geometria de validação ao longo do tempo.",
    "SupCon reduces within-class distance while increasing the ratio between neighboring classes.": "SupCon reduz a distância dentro da classe e aumenta a razão entre classes vizinhas.",
    "A two-dimensional UMAP view.": "Uma visualização UMAP bidimensional.",
    "UMAP is only a projection, but it makes the improved local grouping visible. The quantitative validation metrics remain the basis for model selection.": "UMAP é apenas uma projeção, mas torna visível o melhor agrupamento local. As métricas quantitativas de validação continuam sendo a base da seleção do modelo.",
    "04 · From selfie to match": "04 · Da selfie ao resultado",
    "Three views become one temporary prototype.": "Três visões se tornam um protótipo temporário.",
    "At inference time, the system applies the same detection, cropping, standardization, and normalization used during training. It averages three selfie embeddings and normalizes the result into one more stable user prototype.": "Na inferência, o sistema aplica a mesma detecção, recorte, padronização e normalização do treinamento. Ele calcula a média de três embeddings e normaliza o resultado em um protótipo mais estável.",
    "3 selfies": "3 selfies",
    "processed in memory": "processadas em memória",
    "Mean embedding": "Embedding médio",
    "temporary prototype": "protótipo temporário",
    "Cosine search": "Busca por cosseno",
    "3 nearest prototypes": "3 protótipos mais próximos",
    "Privacy is part of the architecture.": "Privacidade faz parte da arquitetura.",
    "Selfies and the temporary prototype are discarded immediately after matching. They are never written to disk or added to the celebrity database.": "As selfies e o protótipo temporário são descartados após a comparação. Eles nunca são gravados em disco nem adicionados ao banco de celebridades.",
    "Ready for your close-up?": "Pronto para o seu close?",
    "Meet your closest celebrity lookalikes.": "Descubra quais celebridades mais se parecem com você.",
    "Find my celebrity twins": "Encontrar meus sósias famosos",
    "Requesting camera access…": "Solicitando acesso à câmera…",
    "Camera ready. Center one face inside the guide.": "Câmera pronta. Centralize um rosto dentro da guia.",
    "Retake 3 selfies": "Tirar 3 selfies novamente",
    "Use camera": "Usar câmera",
    "Reading the stars…": "Consultando as estrelas…",
    "Matching failed. Please try again.": "Não foi possível comparar. Tente novamente.",
    "The face studio is busy. Please wait a moment and try again.": "O estúdio está ocupado. Aguarde um momento e tente novamente.",
    "Caption copied — paste it with your reveal card.": "Legenda copiada — cole junto com seu card.",
    "We couldn't prepare the celebrity photo.": "Não foi possível preparar a foto da celebridade.",
    "Your red carpet reveal is ready for its audience!": "Sua revelação no tapete vermelho está pronta!",
    "Image downloaded. Add it to your post with the caption below.": "Imagem baixada. Adicione-a à publicação com a legenda abaixo.",
    "Image downloaded — upload it to your favorite social app.": "Imagem baixada — envie-a ao seu aplicativo social favorito.",
    "Creating your red carpet reveal…": "Criando sua revelação no tapete vermelho…",
    "Choose an app. On mobile, the image opens in your share sheet.": "Escolha um aplicativo. No celular, a imagem abrirá no menu de compartilhamento.",
    "We couldn't create the reveal card.": "Não foi possível criar o card de revelação.",
    "Reveal card downloaded.": "Card de revelação baixado.",
    "Copying is unavailable here. Use More to share the caption.": "Não é possível copiar aqui. Use Mais para compartilhar a legenda.",
    "THE RED CARPET REVEAL": "A REVELAÇÃO DO TAPETE VERMELHO",
    "MY CELEBRITY LOOKALIKE IS": "MEU SÓSIA FAMOSO É",
    "TWIN": "SÓSIA"
  };

  const normalize = (locale) => (String(locale).toLowerCase().startsWith("pt") ? "pt-BR" : "en");
  let locale = "en";

  function translate(value) {
    return locale === "pt-BR" ? portuguese[value] || value : value;
  }

  function translateTextNode(node) {
    const original = node.__celebrityTwinEnglish || node.nodeValue.trim();
    if (!original) return;
    node.__celebrityTwinEnglish = original;
    const translated = translate(original);
    const leading = node.nodeValue.match(/^\s*/)[0];
    const trailing = node.nodeValue.match(/\s*$/)[0];
    node.nodeValue = `${leading}${translated}${trailing}`;
  }

  function apply(nextLocale, persist = false) {
    locale = normalize(nextLocale);
    document.documentElement.lang = locale;
    const methodology = document.body.classList.contains("methodology-page");
    document.title = locale === "pt-BR"
      ? methodology ? "Como funciona — Celebrity Twin" : "Celebrity Twin — Qual é o seu sósia famoso?"
      : methodology ? "How it works — Celebrity Twin" : "Celebrity Twin — Who's your famous lookalike?";
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) translateTextNode(walker.currentNode);
    document.querySelectorAll("[data-locale]").forEach((button) => {
      button.setAttribute("aria-pressed", String(normalize(button.dataset.locale) === locale));
    });
    if (persist) localStorage.setItem(STORAGE_KEY, locale);
    window.dispatchEvent(new CustomEvent("celebrity-twin:languagechange", { detail: { locale } }));
  }

  async function init() {
    let selected = localStorage.getItem(STORAGE_KEY);
    if (!selected) {
      try {
        const response = await fetch("/api/config");
        if (response.ok) selected = (await response.json()).suggested_locale;
      } catch (_) {
        selected = navigator.language;
      }
    }
    apply(selected || navigator.language);
    document.querySelectorAll("[data-locale]").forEach((button) => {
      button.addEventListener("click", () => apply(button.dataset.locale, true));
    });
    new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
          else if (node.nodeType === Node.ELEMENT_NODE) {
            const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) translateTextNode(walker.currentNode);
          }
        });
      }
    }).observe(document.body, { childList: true, subtree: true });
  }

  window.CelebrityTwinI18n = { apply, get locale() { return locale; }, t: translate };
  document.addEventListener("DOMContentLoaded", init);
})();
