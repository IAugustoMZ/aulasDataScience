"""
Synthetic safety record generator for FPSO industrial safety classification webinar.
All parameters are read from configs/data.yaml and configs/labels.yaml.
No LLMs used — pure Python string manipulation.
"""

import argparse
import random
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_root() -> Path:
    # src/data/generate.py  ->  ../../  =  project root
    return Path(__file__).parent.parent.parent


# ── Vocabulary banks ──────────────────────────────────────────────────────────
# Each key maps to a list of phrase alternatives.
# All text is pt-BR. The more options per key, the more lexical diversity.

# ── Who reported / discovered ─────────────────────────────────────────────────
REPORTER = [
    "O operador",
    "O técnico de segurança",
    "O supervisor de operações",
    "O inspetor de campo",
    "A equipe de manutenção",
    "O rondeiro noturno",
    "O técnico de instrumentação",
    "O eletricista de plantão",
    "O mecânico de turno",
    "A brigada de emergência",
    "O coordenador de SMS",
    "O mergulhador líder",
    "O operador de convés",
    "O técnico de processo",
    "O gerente de produção",
    "A equipe de inspeção",
    "O analista de segurança",
    "O operador de SCADA",
    "O auxiliar de operações",
    "O chefe de turno",
]

# ── Discovery verb ────────────────────────────────────────────────────────────
# Verbs describe how the reporter communicated the occurrence.
# Adjacent classes intentionally share verbs — severity signal comes from
# what was found, not how it was reported.
DISCOVERY_VERB = {
    "baixo": [
        "identificou",
        "observou",
        "reportou",
        "registrou",
        "constatou",
        "anotou",
        "sinalizou",
        "verificou",
        "detectou",
        "notificou",
        "documentou",
        "apontou",
    ],
    "medio": [
        "identificou",
        "constatou",
        "reportou",
        "detectou",
        "verificou",
        "notificou",
        "registrou",
        "sinalizou",
        "comunicou",
        "informou",
        "alertou",
        "apontou",
    ],
    "alto": [
        "relatou",
        "comunicou",
        "notificou",
        "reportou",
        "registrou",
        "identificou",
        "alertou",
        "informou",
        "constatou",
        "detectou",
        "acionou",
        "sinalizou",
    ],
    "critico": [
        "relatou",
        "comunicou",
        "notificou",
        "reportou",
        "alertou",
        "informou",
        "acionou",
        "registrou",
        "detectou",
        "identificou",
        "constatou",
        "sinalizou",
    ],
}

# ── Occurrence noun (what happened) ──────────────────────────────────────────
# Phrases describe the type of occurrence observed by the reporter.
# Rules:
#   - No class-exclusive severity conclusions ("fatalidade", "emergência máxima",
#     "potencial de lesão grave")
#   - Adjacent classes share some vocabulary; the distinguishing signal comes
#     from the combination of equipment, area, chemical agent, and context
#   - Language: factual observable state, not risk classification
OCCURRENCE_NOUN = {
    "baixo": [
        "uma condição fora do padrão identificada em inspeção",
        "uma falha de documentação operacional",
        "um desvio de organização e limpeza na área",
        "uma condição de uso incorreto de equipamento de proteção",
        "um desvio de procedimento identificado durante auditoria",
        "uma situação de trabalho que requer avaliação técnica",
        "uma ocorrência sem dano imediato registrado",
        "um desvio em checklist pré-operacional",
        "uma condição de iluminação inadequada em área de circulação",
        "uma falha em sinalização de segurança",
        "um item fora de conformidade em inspeção de equipamento",
        "uma condição de exposição a agente físico no posto de trabalho",
        "um desvio identificado em permissão de trabalho",
        "uma falha em barreira de contenção sem vazamento ativo",
    ],
    "medio": [
        "uma situação de trabalho que requer avaliação técnica",
        "um desvio em checklist pré-operacional",
        "uma falha em barreira de contenção sem vazamento ativo",
        "um quase acidente durante execução de tarefa",
        "uma condição de equipamento fora da faixa operacional",
        "um vazamento de pequeno volume em ponto de conexão",
        "uma falha em procedimento de bloqueio e isolamento",
        "uma condição ergonômica inadequada no posto de trabalho",
        "uma situação de exposição a agente nocivo acima do limite de ação",
        "um desvio na permissão de trabalho identificado em campo",
        "uma falha de comunicação entre equipes com impacto na tarefa",
        "uma condição de trabalho em altura com proteção insuficiente",
        "um evento sem lesão registrada mas com potencial de escalada",
        "uma falha de equipamento com impacto na continuidade operacional",
    ],
    "alto": [
        "um quase acidente durante execução de tarefa",
        "uma condição de equipamento fora da faixa operacional",
        "um vazamento de pequeno volume em ponto de conexão",
        "uma condição de trabalho em altura com proteção insuficiente",
        "um evento sem lesão registrada mas com potencial de escalada",
        "uma falha de equipamento com impacto na continuidade operacional",
        "um evento com trabalhadores diretamente expostos à condição",
        "uma falha em barreira primária de segurança do processo",
        "um vazamento de produto pressurizado com ignição possível",
        "uma falha em sistema de proteção coletiva em área de risco",
        "um evento com exposição a fluido quente ou substância nociva",
        "uma condição estrutural com deformação identificada em inspeção",
        "um incidente com trabalhador relatando sintomas de exposição",
        "uma situação com múltiplos trabalhadores próximos ao ponto de falha",
    ],
    "critico": [
        "um evento com trabalhadores diretamente expostos à condição",
        "uma falha em barreira primária de segurança do processo",
        "um vazamento de produto pressurizado com ignição possível",
        "um incidente com trabalhador relatando sintomas de exposição",
        "uma situação com múltiplos trabalhadores próximos ao ponto de falha",
        "um evento com chama ou fumaça visível na área",
        "uma falha estrutural com dano visível a componente crítico",
        "uma condição com alarme de processo em nível máximo",
        "um evento com liberação de gás detectada acima do nível de alarme",
        "uma falha de contenção com produto atingindo área de trabalho",
        "uma condição com múltiplos sistemas de alarme ativos simultaneamente",
        "uma situação de perda de controle de variável de processo crítica",
        "um incidente com equipamento em condição de falha iminente",
        "uma condição com ativação de sistema de proteção automática",
    ],
}

# ── Location prepositions (varied) ───────────────────────────────────────────
LOCATION_PREP = {
    "convés_principal": [
        "no convés principal da plataforma",
        "na área do convés principal",
        "sobre o convés principal",
        "no convés principal (área aberta)",
    ],
    "praça_de_máquinas": [
        "na praça de máquinas",
        "no interior da praça de máquinas",
        "na sala de máquinas da FPSO",
        "no compartimento da praça de máquinas",
    ],
    "módulo_de_processo": [
        "no módulo de processo",
        "na área de processo da plataforma",
        "no trem de processo",
        "na unidade de processamento de gás",
        "no módulo de separação e tratamento",
    ],
    "área_de_utilidades": [
        "na área de utilidades",
        "no módulo de utilidades",
        "na área de geração e utilidades",
        "na sala de utilidades auxiliares",
    ],
    "alojamento": [
        "no alojamento da plataforma",
        "na área habitacional",
        "no bloco de alojamentos",
        "nas instalações de alojamento da FPSO",
    ],
    "sala_de_controle": [
        "na sala de controle",
        "no centro de controle da plataforma",
        "na sala de controle integrado",
        "no CCR (Centro de Controle e Rádio)",
    ],
    "moonpool": [
        "na área do moonpool",
        "na abertura do moonpool",
        "no compartimento do moonpool",
        "na área de operações subsea pelo moonpool",
    ],
    "deck_de_perfuração": [
        "no deck de perfuração",
        "na área de perfuração da FPSO",
        "na sonda de perfuração",
        "no convés de perfuração",
        "na área do BOP e coluna de perfuração",
    ],
    "turret": [
        "no sistema de turret",
        "na área do turret de ancoragem",
        "no turret de amarração da FPSO",
        "na câmara interna do turret",
        "no sistema de ancoragem e turret",
    ],
    "area_offloading": [
        "na área de offloading",
        "durante operação de offloading",
        "no terminal de transferência de óleo",
        "na área de conexão do mangote de offloading",
    ],
    "cais_de_barcos": [
        "no cais de embarque",
        "na área de embarque e desembarque de barcos",
        "no píer de acostagem de embarcações de apoio",
        "na área de boat landing",
    ],
    "topo_das_torres": [
        "no topo das torres",
        "na área do flare e sistema de queima",
        "próximo ao sistema de flare da plataforma",
        "na estrutura elevada do flare boom",
    ],
}

# ── Equipment context (richer alternatives per equipment) ────────────────────
EQUIPMENT_CONTEXT = {
    "guindaste": [
        "durante operação de içamento com guindaste de convés",
        "durante manobra de carga suspensa com guindaste offshore",
        "em içamento com guindaste pedestal",
        "durante operação de carga e descarga com guindaste",
        "em operação de guindaste com vento acima do limite",
    ],
    "talha_manual": [
        "durante uso de talha manual de corrente",
        "em içamento com talha manual em espaço confinado",
        "durante operação com talha manual de alavanca (tirfor)",
        "no uso de talha manual acima da capacidade nominal",
    ],
    "talha_eletrica": [
        "durante operação de talha elétrica monovia",
        "em içamento com talha elétrica de ponte rolante",
        "durante manobra com talha elétrica de pórtico",
        "em manutenção de talha elétrica com carga suspensa",
    ],
    "andaime": [
        "durante montagem de andaime tubular",
        "em trabalho sobre andaime suspenso",
        "durante desmontagem de andaime em altura",
        "no uso de andaime fachadeiro sem guarda-corpo",
        "em andaime tipo balancim motorizado",
    ],
    "cesto_de_içamento": [
        "durante içamento de pessoal em cesto offshore",
        "em transferência de pessoal por cesto Billy Pugh",
        "durante operação de embarque/desembarque por cesto",
        "em içamento de pessoal com cesto com vento excessivo",
    ],
    "munck": [
        "durante operação de guindaste munck",
        "em içamento com caminhão munck sobre convés",
        "durante manobra de carga com munck em área restrita",
    ],
    "painel_eletrico": [
        "em painel elétrico de distribuição de baixa tensão",
        "durante intervenção em painel MCC (Motor Control Center)",
        "em painel elétrico de média tensão (6,6kV)",
        "durante manutenção em painel de distribuição de instrumentação",
        "em abertura de painel elétrico sem desenergização",
    ],
    "transformador": [
        "próximo a transformador de potência de alta tensão",
        "durante manutenção em transformador de distribuição",
        "em inspeção de transformador com vazamento de óleo isolante",
        "durante comissionamento de transformador de força",
    ],
    "gerador": [
        "no gerador principal de emergência",
        "durante partida de gerador diesel de emergência",
        "em manutenção preventiva de gerador de energia",
        "durante inspeção em gerador a gás em operação",
        "em gerador com defeito no sistema de proteção",
    ],
    "cabo_de_alta_tensao": [
        "em cabo de energia de alta tensão exposto",
        "durante manuseio de cabo de alta tensão sem isolação",
        "em intervenção em cabo de alimentação de 13,8kV",
        "próximo a cabo de alta tensão com isolamento danificado",
    ],
    "subestacao": [
        "na subestação elétrica principal da FPSO",
        "durante manutenção em subestação de média tensão",
        "em intervenção na subestação sem isolamento adequado",
        "na subestação com presença de arco elétrico",
    ],
    "aterramento": [
        "no sistema de aterramento elétrico da plataforma",
        "durante verificação de continuidade de aterramento",
        "em cabo de aterramento de equipamento rotativo",
        "no aterramento de tanque de armazenamento de óleo",
    ],
    "sensor_de_gas": [
        "em sensor fixo de detecção de gás inflamável",
        "durante calibração de sensor de H₂S",
        "em sistema de detecção de gás com alarme ativo",
        "durante substituição de sensor de CO₂ em espaço confinado",
        "em sensor de gás com leitura acima do nível de alarme",
    ],
    "detector_de_chama": [
        "em detector de chama UV/IR em área classificada",
        "durante manutenção de detector de chama de processo",
        "em teste funcional de detector de chama com falha",
        "durante inspeção de detector de chama com lente obstruída",
    ],
    "valvula_de_controle": [
        "em válvula de controle de fluxo de processo",
        "durante intervenção em válvula de controle pressurizada",
        "em válvula de controle com atuador pneumático defeituoso",
        "durante manutenção de válvula de controle de gás",
        "em válvula de controle com falha de fechamento",
    ],
    "transmissor_de_pressao": [
        "em transmissor de pressão de linha de processo",
        "durante substituição de transmissor de pressão diferencial",
        "em calibração de transmissor com sistema pressurizado",
        "no transmissor de pressão com leitura espúria",
    ],
    "transmissor_de_nivel": [
        "em transmissor de nível de tanque de processo",
        "durante manutenção em transmissor de nível radar",
        "em transmissor de nível com falha de comunicação HART",
        "no medidor de nível com leitura incorreta causando transbordamento",
    ],
    "sistema_scada": [
        "no sistema SCADA de supervisão e controle",
        "durante falha crítica no sistema de controle distribuído (DCS)",
        "em intervenção no sistema SCADA com inibição de alarmes",
        "durante atualização de software no sistema de controle",
    ],
    "bomba_centrifuga": [
        "em bomba centrífuga de transferência de processo",
        "durante partida de bomba centrífuga de alta pressão",
        "em bomba centrífuga com selo mecânico com vazamento",
        "durante manutenção de bomba centrífuga com fluido quente",
        "em bomba centrífuga com vibração acima do limite de alarme",
    ],
    "compressor": [
        "no compressor de gás de alta pressão",
        "durante partida de compressor reciprocante de gás de exportação",
        "em compressor centrífugo com vibração elevada",
        "durante manutenção de compressor de gás lift",
        "no compressor com alarme de alta temperatura de descarga",
    ],
    "turbina_a_gas": [
        "na turbina a gás de geração de energia",
        "durante comissionamento de turbina a gás de acionamento",
        "em turbina a gás com vibração excessiva em rotor",
        "durante inspeção interna de turbina a gás com hot-section danificada",
    ],
    "motor_diesel": [
        "no motor diesel auxiliar de emergência",
        "durante manutenção de motor diesel de bomba de incêndio",
        "em motor diesel com superaquecimento de sistema de arrefecimento",
        "no motor diesel com emissão de fumaça preta excessiva",
    ],
    "ventilador": [
        "no ventilador de exaustão de área classificada",
        "durante manutenção de ventilador de insuflamento de praça de máquinas",
        "em ventilador com falha de proteção de motor",
        "no ventilador com pás desequilibradas e vibração elevada",
    ],
    "agitador": [
        "no agitador de tanque de tratamento químico",
        "durante manutenção de agitador de vaso de processo",
        "em agitador com vedação mecânica com vazamento de produto",
    ],
    "separador": [
        "no separador trifásico de produção",
        "durante intervenção no separador de alta pressão",
        "em separador de produção com nível alto crítico",
        "no separador com transbordamento de líquido para linha de gás",
        "durante abertura de separador com pressão residual",
    ],
    "vaso_flash": [
        "no vaso de flash de gás de baixa pressão",
        "durante inspeção em vaso de flash com pressão residual",
        "em vaso de flash com válvula de segurança atuando",
    ],
    "trocador_de_calor": [
        "no trocador de calor de processo",
        "durante manutenção de trocador de placas com produto quente",
        "em trocador casco-tubo com vazamento de fluido quente",
        "durante limpeza química de trocador de calor pressurizado",
    ],
    "tubulacao_processo": [
        "em tubulação de processo de alta pressão",
        "durante intervenção em linha de processo pressurizada",
        "em tubulação com corrosão avançada e risco de ruptura",
        "durante manutenção em tubulação com fluido quente sob pressão",
        "em flange de tubulação de gás com vazamento detectado",
    ],
    "mangote_de_transferencia": [
        "no mangote de transferência de óleo de offloading",
        "durante conexão de mangote de offloading em mar agitado",
        "em mangote com pressão acima do limite de operação",
        "durante desconexão de mangote com produto ainda pressurizado",
    ],
    "manifold": [
        "no manifold de produção de poços",
        "durante intervenção em manifold pressurizado",
        "em manifold com válvula de isolamento travada aberta",
        "no manifold de injeção de água com vazamento em flange",
    ],
    "deck_estrutural": [
        "na estrutura metálica do deck principal",
        "em viga estrutural com corrosão avançada",
        "durante inspeção de integridade estrutural do casco",
        "em área de deck com deformação estrutural identificada",
    ],
    "suporte_de_tubulacao": [
        "em suporte de tubulação com falha estrutural",
        "durante inspeção de suportes de tubulação de processo",
        "em suporte de mola com perda de tensão",
        "no suporte de tubulação de alta temperatura com deformação",
    ],
    "escada_marinheiro": [
        "em escada tipo marinheiro sem proteção de gaiola",
        "durante subida em escada marinheiro com superfície molhada",
        "em escada marinheiro com degrau danificado",
        "durante descida de escada marinheiro com materiais na mão",
    ],
    "guarda_corpo": [
        "em sistema de guarda-corpo com componente faltando",
        "durante reparo de guarda-corpo danificado em altura",
        "em guarda-corpo com solda fraturada identificada",
        "no guarda-corpo com fixação insuficiente",
    ],
    "porta_corta_fogo": [
        "em porta corta-fogo com mecanismo de fechamento defeituoso",
        "durante manutenção de porta corta-fogo em corredor de fuga",
        "em porta corta-fogo travada aberta em área de risco",
        "no sistema de porta corta-fogo com falha de sinalização",
    ],
    "casco": [
        "no casco da FPSO com corrosão estrutural identificada",
        "durante inspeção subaquática do casco da embarcação",
        "em compartimento do casco com entrada de água detectada",
        "no casco com dano por colisão com embarcação de apoio",
    ],
    "extintor": [
        "em extintor de incêndio com pressão abaixo do mínimo",
        "durante uso de extintor em combate a princípio de incêndio",
        "em extintor com lacre violado e carga comprometida",
        "no extintor com prazo de validade vencido",
    ],
    "sprinkler": [
        "no sistema de sprinklers com bico obstruído",
        "durante teste hidrostático do sistema de sprinklers",
        "em sprinkler com acionamento não intencional",
        "no sistema de chuveiros automáticos com falha de pressão",
    ],
    "detector_de_fumaca": [
        "em detector de fumaça com alarme falso recorrente",
        "durante manutenção de detector de fumaça em área habitável",
        "em detector de fumaça com câmara de ionização contaminada",
        "no detector de fumaça instalado em ambiente com névoa de óleo",
    ],
    "painel_de_alarme": [
        "no painel de alarme de incêndio com falha de comunicação",
        "durante manutenção no PABFI (Painel de Alarme e Brigada de Incêndio)",
        "em painel de alarme com múltiplos pontos em falha",
        "no painel de controle de incêndio com bateria de backup descarregada",
    ],
    "sistema_diluvio": [
        "no sistema de dilúvio de área de processo",
        "durante acionamento intempestivo do sistema de dilúvio",
        "em sistema de dilúvio com bocais obstruídos",
        "durante teste do sistema de dilúvio com falha de pressão",
    ],
    "mangueira_incendio": [
        "em mangueira de combate a incêndio com rasgos",
        "durante desenrolamento de mangueira de incêndio em emergência",
        "em mangueira de incêndio com conexão com vazamento",
        "no carretel de mangueira de incêndio com enrolamento irregular",
    ],
    "umbilical": [
        "no umbilical submarino com dano externo identificado",
        "durante inspeção de umbilical em ROV survey",
        "em umbilical com falha de continuidade elétrica",
        "durante lançamento de umbilical com torção excessiva",
    ],
    "ROV": [
        "durante operação de ROV em inspeção submarina",
        "em operação de ROV com perda de comunicação umbilical",
        "durante intervenção subsea com ROV de trabalho (WROV)",
        "em operação de ROV com colisão com estrutura submarina",
    ],
    "sistema_saturation_diving": [
        "durante mergulho em saturação em módulo de habitação",
        "em operação de saturação com temperatura de câmara fora do padrão",
        "durante descompressão de sistema de mergulho em saturação",
        "em operação de saturação com falha no sistema de suprimento de gás",
    ],
    "boia_de_sinalizacao": [
        "em boia de sinalização submarina com perda de posicionamento",
        "durante manutenção de boia de ancoragem de linha submarina",
        "em boia de sinalização de pipeline com dano por colisão",
    ],
    "ancora": [
        "no sistema de ancoragem da FPSO com linha rompida",
        "durante inspeção de âncoras e correntes de amarração",
        "em linha de ancoragem com corrosão avançada identificada",
        "no sistema de amarração com excursão de posicionamento dinâmico",
    ],
    "risers": [
        "no sistema de risers flexíveis de produção",
        "durante inspeção de riser com dano em camada de armadura",
        "em riser de injeção com vazamento no conector de topo",
        "no riser de gás com vibração induzida por corrente (VIV)",
        "durante intervenção em riser com pressão de processo ativa",
    ],
}

# ── Immediate field action taken by the reporter ─────────────────────────────
# These are the actions taken by the person filing the report, written as they
# would appear in a field log at the moment of reporting — NOT the final
# investigation outcome or management decision.
#
# Design rules:
#   - Same action types can appear across adjacent classes (e.g., "communicated
#     to supervisor", "isolated the area") — severity emerges from the *degree*
#     of action, not from class-exclusive vocabulary
#   - No conclusive severity language ("total evacuation", "fatal victim")
#   - ~30 % vocabulary overlap between adjacent pairs
CONSEQUENCE = {
    "baixo": [
        "O relator comunicou a supervisão e registrou a ocorrência no sistema.",
        "A condição foi anotada no livro de ocorrências do turno.",
        "O relator sinalizou a área e informou o encarregado responsável.",
        "A ocorrência foi registrada para acompanhamento pela equipe de SMS.",
        "O relator orientou o trabalhador envolvido e documentou o evento.",
        "A condição foi identificada, anotada e encaminhada para análise.",
        "O equipamento foi marcado para inspeção e a supervisão informada.",
        "O relator emitiu cartão de anomalia e aguarda avaliação técnica.",
        "A área foi sinalizada e a ocorrência inserida na ordem do dia.",
        "O relator documentou o evento e aguarda definição de ação corretiva.",
        "A supervisão foi comunicada verbalmente durante a passagem de turno.",
        "O relator preencheu o formulário de ocorrência e entregou ao SMS.",
        "O equipamento foi colocado em modo de atenção para monitoramento.",
        "A condição foi anotada e incluída no plano de inspeção do próximo turno.",
        "O relator interrompeu a tarefa e consultou o supervisor de operações.",
    ],
    "medio": [
        "O relator interrompeu a tarefa e consultou o supervisor de operações.",
        "A área foi sinalizada e a supervisão notificada para avaliação.",
        "O relator isolou o ponto e acionou a equipe de manutenção.",
        "A atividade foi pausada e o encarregado chamado ao local.",
        "O relator coletou evidências e registrou as condições no formulário.",
        "O trabalhador envolvido foi afastado da área e a supervisão avisada.",
        "A equipe do turno foi comunicada e o acesso à área restringido.",
        "O relator solicitou inspeção técnica antes de retomar a atividade.",
        "A permissão de trabalho foi revisada e a equipe reagrupada.",
        "O relator acionou o técnico de segurança para avaliação in loco.",
        "A área foi sinalizada e o equipamento bloqueado preventivamente.",
        "O relator comunicou o ocorrido ao chefe de turno imediatamente.",
        "A tarefa foi suspensa até novo procedimento ser emitido.",
        "O relator fez medição com detector portátil e registrou os valores.",
        "A supervisão foi acionada e o equipamento desligado para inspeção.",
    ],
    "alto": [
        "O relator acionou o supervisor imediatamente e isolou a área.",
        "A área foi interditada e a equipe afastada do ponto de risco.",
        "O relator comunicou o chefe de turno e acionou o técnico de SMS.",
        "A permissão de trabalho foi cancelada e a área bloqueada.",
        "O relator solicitou reforço de equipe e realizou medição no local.",
        "A brigada foi notificada e o acesso controlado até nova avaliação.",
        "O relator fez medição com detector portátil e registrou os valores.",
        "A supervisão foi acionada e o equipamento desligado para inspeção.",
        "O relator reuniu a equipe do turno e repassou as condições do local.",
        "A área foi demarcada com barreira física e o gestor de turno informado.",
        "O relator acionou manutenção de emergência e acompanhou intervenção.",
        "O trabalhador foi encaminhado à enfermaria para avaliação.",
        "A operação foi interrompida localmente e o supervisor notificado.",
        "O relator fotografou a condição e enviou para avaliação técnica.",
        "A equipe foi reposicionada e o supervisor de SMS chamado ao local.",
    ],
    "critico": [
        "O relator acionou o supervisor imediatamente e isolou a área.",
        "A área foi interditada e a equipe afastada do ponto de risco.",
        "O relator comunicou o chefe de turno e acionou o técnico de SMS.",
        "A brigada foi notificada e o acesso controlado até nova avaliação.",
        "O trabalhador foi encaminhado à enfermaria para avaliação.",
        "A operação foi interrompida localmente e o supervisor notificado.",
        "O relator fotografou a condição e enviou para avaliação técnica.",
        "O relator acionou o alarme da área e retirou a equipe imediatamente.",
        "A brigada de emergência foi acionada e o perímetro estabelecido.",
        "O relator comunicou a sala de controle e aguardou instruções.",
        "A equipe de resposta foi mobilizada e o acesso à área bloqueado.",
        "O relator registrou as condições no formulário e aguarda equipe técnica.",
        "O supervisor de turno foi acionado e a área isolada preventivamente.",
        "O relator afastou os trabalhadores e acionou o responsável de turno.",
        "A equipe foi evacuada da área e o responsável de SMS convocado.",
    ],
}

# ── Observational context fragments injected in narrative body ───────────────
# Phrases describe what the reporter *saw or measured* at the moment of the
# occurrence — NOT the severity conclusion. Adjacent classes share vocabulary
# to create genuine lexical ambiguity that forces models to rely on weaker
# contextual signals rather than class-exclusive keywords.
#
# Design rules:
#   - No class-exclusive severity labels ("fatalidade", "emergência máxima")
#   - ~40 % overlap between adjacent classes (baixo↔medio, medio↔alto, alto↔critico)
#   - Language: measurable observations, equipment state, worker perception
SEVERITY_CONTEXT = {
    "baixo": [
        "no momento da inspeção de rotina",
        "durante ronda de verificação do turno",
        "identificado durante checklist pré-operacional",
        "observado no início da atividade programada",
        "constatado durante inspeção visual do equipamento",
        "notado durante passagem de turno",
        "verificado em auditoria de campo",
        "registrado durante manutenção preventiva programada",
        "detectado em condição que exigiu avaliação técnica",
        "observado com equipamento fora do estado esperado",
    ],
    "medio": [
        "constatado durante inspeção visual do equipamento",
        "notado durante passagem de turno",
        "detectado em condição que exigiu avaliação técnica",
        "observado com equipamento fora do estado esperado",
        "identificado com sinalização de alarme ativo no painel",
        "verificado com medição acima do limite de referência",
        "percebido com ruído anormal no equipamento",
        "constatado com vazamento visível no ponto de inspeção",
        "registrado após relato de trabalhador da área",
        "identificado com odor característico na região",
    ],
    "alto": [
        "identificado com sinalização de alarme ativo no painel",
        "verificado com medição acima do limite de referência",
        "percebido com ruído anormal no equipamento",
        "constatado com vazamento visível no ponto de inspeção",
        "identificado com odor característico na região",
        "observado com múltiplos trabalhadores presentes na área",
        "constatado com pressão do sistema fora da faixa operacional",
        "verificado com temperatura acima do setpoint de alarme",
        "detectado com leitura de detector portátil acima do nível de ação",
        "registrado com trabalhador relatando sintomas de exposição",
    ],
    "critico": [
        "observado com múltiplos trabalhadores presentes na área",
        "constatado com pressão do sistema fora da faixa operacional",
        "verificado com temperatura acima do setpoint de alarme",
        "detectado com leitura de detector portátil acima do nível de ação",
        "registrado com trabalhador relatando sintomas de exposição",
        "constatado com alarme geral da plataforma acionado",
        "verificado com chama visível ou fumaça densa na área",
        "observado com vazamento de fluido sob alta pressão",
        "identificado com estrutura com deformação visível a olho nu",
        "detectado com múltiplos alarmes simultâneos no DCS",
    ],
}

# ── Supplementary context fragments ──────────────────────────────────────────
WEATHER_CONDITION = [
    "com mar estado 4 e ventos de 25 nós",
    "em condições de chuva intensa e visibilidade reduzida",
    "com temperatura ambiente de 38°C no convés",
    "em condições de neblina e umidade elevada",
    "com ondas de 3,5m e vento nordeste de 30 nós",
    "em período noturno com iluminação artificial",
    "com mar calmo e condições meteorológicas favoráveis",
    "em condições de ventania com rajadas acima de 40 nós",
]

WORKER_STATE = [
    "após 10 horas consecutivas de trabalho",
    "durante execução de hora extra no final de turno",
    "com equipe reduzida por afastamento de colaborador",
    "durante período de startup após parada programada",
    "em tarefa não rotineira sem análise de risco prévia",
    "com trabalhador em primeiro dia na função",
    "durante execução simultânea de múltiplas tarefas críticas",
    "com equipamento de proteção individual incompleto",
    "após modificação de processo não comunicada ao turno",
    "durante operação com permissão de trabalho vencida",
]

NORM_REFERENCE = [
    "Ocorrência registrada conforme procedimento NR-10 de segurança elétrica.",
    "Notificação emitida em conformidade com os requisitos da NR-33.",
    "Evento documentado seguindo os critérios da NR-35 para trabalho em altura.",
    "Registro realizado de acordo com o plano de gerenciamento de riscos da unidade.",
    "Ocorrência documentada conforme análise HAZOP do módulo afetado.",
    "Evento registrado conforme procedimento SMS vigente da operadora.",
    "Notificação enviada conforme NORMAM-01 da Marinha do Brasil.",
    "Ocorrência comunicada conforme requisitos da NR-37 para plataformas offshore.",
    "Relatório emitido conforme exigência da ANP para acidentes offshore.",
    "Incidente classificado e registrado conforme matriz de criticidade da unidade.",
]

# ── Sentence structure templates ──────────────────────────────────────────────
# Template slots:  {reporter} {discovery_verb} {occurrence_noun} {location} {equipment_ctx}
# Optional extras: {weather}, {worker_state}, {consequence}, {norm}
# There are 4 different sentence structures to vary syntax further.

SENTENCE_STRUCTURES = [
    "struct_standard",      # Reporter + verb + noun + location + equipment
    "struct_passive",       # Noun (passive) + location + equipment + reporter
    "struct_location_first",# Location + equipment + reporter + verb + noun
    "struct_brief",         # Reporter + verb + noun + location (no equipment context)
]


def build_narrative(
    risk_class: str,
    area_id: str,
    equip_subclass: str,
    chemical_id: str,
    fator: str,
    turno: str,
    include_chemical: bool,
    include_norm: bool,
    include_time: bool,
    include_weather: bool,
    include_worker_state: bool,
    rng: random.Random,
    location_prep: dict,
    equipment_context: dict,
    chemical_desc: dict,
    fator_desc: dict,
    turno_desc: dict,
) -> str:
    reporter = rng.choice(REPORTER)
    verb = rng.choice(DISCOVERY_VERB[risk_class])
    noun = rng.choice(OCCURRENCE_NOUN[risk_class])
    location = rng.choice(location_prep.get(area_id, [f"na área {area_id}"]))
    equip_ctx = rng.choice(equipment_context.get(equip_subclass, [f"no equipamento {equip_subclass}"]))
    consequence = rng.choice(CONSEQUENCE[risk_class])
    # Frente 1: severity cue always injected in body so BERT sees it early
    severity_cue = rng.choice(SEVERITY_CONTEXT[risk_class])

    structure = rng.choice(SENTENCE_STRUCTURES)

    if structure == "struct_standard":
        sentence = f"{reporter} {verb} {noun} {location}, {equip_ctx}"
    elif structure == "struct_passive":
        noun_upper = noun[0].upper() + noun[1:] if noun else noun
        sentence = f"{noun_upper} foi verificada {location}, {equip_ctx}. {reporter.capitalize()} realizou o registro"
    elif structure == "struct_location_first":
        sentence = f"{location.capitalize()}, {equip_ctx}, {reporter.lower()} {verb} {noun}"
    else:  # struct_brief
        sentence = f"{reporter} {verb} {noun} {location}"

    # Severity cue goes right after the opening clause, before optional extras
    sentence += f", {severity_cue}"

    extras = []

    if include_time:
        turno_ctx = rng.choice(turno_desc.get(turno, [f"no turno {turno}"]))
        extras.append(turno_ctx)

    if include_weather:
        extras.append(rng.choice(WEATHER_CONDITION))

    if include_worker_state:
        extras.append(rng.choice(WORKER_STATE))

    fator_ctx = fator_desc.get(fator, "")
    if fator_ctx:
        extras.append(fator_ctx)

    if include_chemical and chemical_id != "nenhum":
        chem_ctx = chemical_desc.get(chemical_id, "")
        if chem_ctx:
            extras.append(chem_ctx)

    if extras:
        sentence += ", " + ", ".join(extras)

    sentence += ". " + consequence

    if include_norm:
        sentence += " " + rng.choice(NORM_REFERENCE)

    return sentence


# ── Chemical descriptions (richer) ───────────────────────────────────────────
CHEMICAL_DESC = {
    "oleo_cru": [
        "com presença de óleo cru na área",
        "com derramamento de óleo cru no convés",
        "com acúmulo de óleo cru bruto em piso",
        "com contato de óleo cru com superfície quente",
    ],
    "gas_natural": [
        "com detecção de gás natural inflamável",
        "com concentração de gás natural acima do LEL",
        "com alarme de gás natural ativo no painel",
        "com presença de hidrocarboneto gasoso na atmosfera",
    ],
    "h2s": [
        "com concentração de H₂S acima do limite de exposição (10 ppm)",
        "com alarme de H₂S ativo no detector portátil",
        "com H₂S detectado acima de 50 ppm (IDLH)",
        "com presença de gás sulfídrico no ambiente confinado",
        "com leitura de H₂S acima do nível de evacuação",
    ],
    "metanol": [
        "com vazamento de metanol em linha de injeção",
        "com derramamento de metanol no convés",
        "com exposição cutânea a metanol sem EPI adequado",
    ],
    "glicol": [
        "com derramamento de monoetilenoglicol (MEG) no convés",
        "com vazamento de MEG em circuito de desidratação de gás",
        "com exposição a glicol em alta temperatura",
    ],
    "hipoclorito": [
        "com exposição a hipoclorito de sódio sem EPC",
        "com derramamento de solução clorada em área de utilidades",
        "com contato de hipoclorito com pele sem EPI adequado",
    ],
    "diesel": [
        "com derramamento de diesel marítimo no convés",
        "com vazamento de diesel em linha de alimentação de motor",
        "com acúmulo de diesel em bacia de contenção",
    ],
    "nitrogenio": [
        "com risco de asfixia por nitrogênio em espaço confinado",
        "com atmosfera deficiente em oxigênio por purga com N₂",
        "com uso de N₂ em ambiente sem monitoramento de O₂",
    ],
    "amonia": [
        "com detecção de amônia acima do limite de tolerância (20 ppm)",
        "com vazamento de NH₃ em sistema de refrigeração",
        "com concentração de amônia exigindo uso de SCBA",
    ],
    "acido_sulfurico": [
        "com exposição a ácido sulfúrico concentrado",
        "com derramamento de H₂SO₄ em área de tratamento",
        "com contato de ácido sulfúrico com EPI inadequado",
    ],
    "nenhum": [""],
}

FATOR_RISCO_DESC = {
    "ruido": [
        "com exposição a ruído acima de 85 dB(A) sem protetor auricular",
        "em ambiente com ruído ocupacional acima dos limites da NR-15",
        "com nível de pressão sonora de 92 dB(A) medido no posto de trabalho",
    ],
    "vibracao": [
        "com vibração de corpo inteiro acima do limite de ação",
        "com vibração mecânica em mãos e braços acima do VLE",
        "com exposição a vibração de equipamento rotativo sem amortecimento",
    ],
    "calor": [
        "em ambiente com temperatura de bulbo úmido acima de 28°C",
        "com IBUTG acima do limite para atividade pesada",
        "em ambiente com temperatura operativa acima de 38°C no convés",
    ],
    "radiacao_nao_ionizante": [
        "com exposição à radiação UV solar sem proteção adequada",
        "com exposição à radiação de arco de solda sem anteparo",
        "com irradiância acima do limite para exposição ocupacional",
    ],
    "pressao_anormal": [
        "em ambiente de pressão hiperbárica durante mergulho",
        "em vaso com pressão acima da PMTA sem revisão de certificado",
        "em sistema pressurizado acima de 150 PSI sem alívio de pressão",
    ],
    "frio": [
        "em ambiente de baixa temperatura sem equipamento de proteção contra frio",
        "com exposição a temperatura abaixo de 4°C sem EPI térmico",
        "em sala de refrigeração sem controle de temperatura corporal",
    ],
    "gas_toxico": [
        "com presença de gases tóxicos no ambiente acima do TLV-TWA",
        "com concentração de CO acima de 25 ppm no ambiente de trabalho",
        "com mistura de gases tóxicos detectada por monitor multigas",
    ],
    "vapores_inflamaveis": [
        "com vapores inflamáveis detectados acima de 10% do LEL",
        "com concentração de vapores de hidrocarboneto no ar",
        "com alarme de LEL ativo em detector fixo de área",
    ],
    "particulas_em_suspensao": [
        "com partículas em suspensão acima do limite de tolerância",
        "com poeira mineral acima de 2 mg/m³ respirável",
        "com névoa de óleo acima do TLV-TWA",
    ],
    "neblinas_acidas": [
        "com neblinas ácidas presentes acima do limite de tolerância",
        "com névoa de ácido clorídrico no ambiente",
        "com neblinas corrosivas sem exaustão local adequada",
    ],
    "poeiras_minerais": [
        "com poeiras minerais em concentração acima do limite",
        "com sílica cristalina detectada acima de 0,1 mg/m³",
        "com poeira de limpeza abrasiva sem controle de engenharia",
    ],
    "microorganismos": [
        "com presença de microorganismos em sistema de água de resfriamento",
        "com Legionella detectada em sistema de HVAC",
        "com contaminação microbiológica em tanque de agua potável",
    ],
    "insetos": [
        "com infestação de insetos em área de alimentos",
        "com presença de insetos em equipamentos elétricos",
    ],
    "fungos": [
        "com contaminação fúngica em sistema de ventilação",
        "com fungos em área habitável com umidade elevada",
    ],
    "postura_inadequada": [
        "com adoção de postura ergonomicamente inadequada por período prolongado",
        "com flexão de tronco acima de 60° por mais de 2 horas",
        "com trabalho em postura forçada sem revezamento",
    ],
    "esforco_repetitivo": [
        "com movimentos repetitivos de membro superior acima de 30 ciclos/min",
        "com esforço repetitivo sem pausas de recuperação adequadas",
        "com ciclo de trabalho repetitivo por período superior a 4 horas contínuas",
    ],
    "levantamento_manual_de_carga": [
        "com levantamento manual de carga acima de 23 kg sem auxílio mecânico",
        "com manuseio de carga pesada em postura inadequada",
        "com transporte manual de carga acima do limite da NR-17",
    ],
    "jornada_prolongada": [
        "após jornada de trabalho de 14 horas consecutivas",
        "com trabalhador em sobrecarga cognitiva por jornada estendida",
        "com fadiga acumulada por turno noturno de 12 horas",
    ],
    "queda_de_altura": [
        "com risco de queda de altura superior a 2 metros sem talabarte",
        "com trabalho em altura sem sistema de proteção coletiva (SPC)",
        "com ausência de linha de vida em tarefa acima de 4 metros",
    ],
    "choque_eletrico": [
        "com risco de choque elétrico por equipamento desenergizado incorretamente",
        "com exposição a partes vivas de equipamento elétrico energizado",
        "com risco de arco elétrico em distância inferior à distância segura",
    ],
    "aprisionamento": [
        "com risco de aprisionamento de membro em ponto de aperto",
        "com risco de enrolamento em parte rotativa sem proteção",
        "com risco de aprisionamento entre carga e estrutura fixa",
    ],
    "projecao_de_fragmentos": [
        "com projeção de fragmentos metálicos durante atividade de corte",
        "com risco de projeção de partículas abrasivas sem anteparo",
        "com risco de projeção de fluido pressurizado por falha de flange",
    ],
    "atropelamento": [
        "com risco de atropelamento por veículo de carga no convés",
        "com circulação de veículo pesado em área sem sinalização de pedestres",
        "com risco de colisão entre veículo e trabalhador a pé",
    ],
    "queimadura": [
        "com risco de queimadura térmica por superfície quente acima de 43°C",
        "com exposição a fluido quente acima de 60°C em trabalho de manutenção",
        "com risco de queimadura química por produto corrosivo sem EPI adequado",
    ],
}

# ── Time-of-day generation ────────────────────────────────────────────────────
# Turno A: 07h-19h  |  Turno B: 19h-07h  |  Turno C: passagem (06:45-07:15 / 18:45-19:15)
# Weights model: first-hour spike, mid-shift plateau, pre-handover spike.
# critico/alto: heavier weight on handover window and first hour.
# baixo: heavier weight on final 30 min (housekeeping paperwork before handover).

_TZ_BRASIL = ZoneInfo("America/Sao_Paulo")

# Each entry: (hour_start, minute_start, duration_minutes, weight)
_TURNO_A_SLOTS: list[tuple[int, int, int, float]] = [
    (7, 0, 120, 2.5),    # first 2h: PTW / entry inspection spike
    (9, 0, 180, 1.0),    # mid-shift plateau
    (12, 0, 120, 0.8),   # post-lunch dip
    (14, 0, 180, 1.0),   # afternoon plateau
    (17, 30, 90, 2.0),   # pre-handover rush
]

_TURNO_B_SLOTS: list[tuple[int, int, int, float]] = [
    (19, 0, 120, 2.5),   # first 2h: entry spike
    (21, 0, 180, 1.0),   # early-night plateau
    (0, 0, 180, 0.8),    # overnight fatigue dip
    (3, 0, 120, 1.0),    # late-night plateau
    (5, 30, 90, 2.0),    # pre-handover rush
]

# Passagem: tight window around shift change
_TURNO_C_SLOTS: list[tuple[int, int, int, float]] = [
    (6, 45, 30, 1.0),
    (18, 45, 30, 1.0),
]

# critico/alto get extra weight on handover slots (index -1 of A and B, and all C)
_HIGH_RISK_EXTRA_WEIGHT = 2.0


def _sample_time_in_slot(
    hour_start: int,
    minute_start: int,
    duration_minutes: int,
    rng: random.Random,
) -> tuple[int, int]:
    offset = rng.randint(0, duration_minutes - 1)
    total_minutes = hour_start * 60 + minute_start + offset
    total_minutes %= 1440  # wrap around midnight
    return total_minutes // 60, total_minutes % 60


def sample_time_of_day(
    turno: str,
    risk_class: str,
    rng: random.Random,
) -> tuple[int, int]:
    """Return (hour, minute) sampled from realistic shift distribution."""
    is_high_risk = risk_class in ("critico", "alto")

    if turno == "A":
        slots = _TURNO_A_SLOTS
        weights = [w * (_HIGH_RISK_EXTRA_WEIGHT if (is_high_risk and i == len(slots) - 1) else 1.0)
                   for i, (_, _, _, w) in enumerate(slots)]
    elif turno == "B":
        slots = _TURNO_B_SLOTS
        weights = [w * (_HIGH_RISK_EXTRA_WEIGHT if (is_high_risk and i == len(slots) - 1) else 1.0)
                   for i, (_, _, _, w) in enumerate(slots)]
    else:  # C — passagem de turno
        slots = _TURNO_C_SLOTS
        weights = [w for _, _, _, w in slots]

    h_start, m_start, dur, _ = rng.choices(slots, weights=weights, k=1)[0]
    return _sample_time_in_slot(h_start, m_start, dur, rng)


TURNO_DESC = {
    "A": [
        "durante o turno A (07h-19h)",
        "no turno diurno (A)",
        "no início do turno da manhã",
        "ao final do turno diurno",
        "durante o turno A próximo à passagem",
    ],
    "B": [
        "durante o turno B (19h-07h)",
        "no turno noturno (B)",
        "no início do turno da noite",
        "ao final do turno noturno",
        "durante o turno B com equipe reduzida",
    ],
    "C": [
        "durante a passagem de turno entre A e B",
        "no período de troca de turno",
        "durante handover de turno com sobreposição de equipes",
        "na transição entre os turnos A e B",
    ],
}


# ── Class sampling ─────────────────────────────────────────────────────────────

def sample_class(distribution: dict, rng: random.Random) -> str:
    classes = list(distribution.keys())
    weights = [distribution[c] for c in classes]
    return rng.choices(classes, weights=weights, k=1)[0]


def adjacent_class(risk_class: str, ordered: list, rng: random.Random) -> str:
    idx = ordered.index(risk_class)
    candidates = []
    if idx > 0:
        candidates.append(ordered[idx - 1])
    if idx < len(ordered) - 1:
        candidates.append(ordered[idx + 1])
    return rng.choice(candidates) if candidates else risk_class


# ── Frente 2: area weights per class ─────────────────────────────────────────
# risk_weight from labels.yaml amplified per class. critico/alto concentrate
# in high-risk areas; baixo/medio in lower-risk areas. Target Cramér's V ~0.08-0.12.
_AREA_CLASS_MULTIPLIERS: dict[str, dict[str, float]] = {
    # area_id → {class → extra multiplier on top of base risk_weight}
    "deck_de_perfuração":   {"critico": 3.0, "alto": 2.0, "medio": 1.0, "baixo": 0.4},
    "módulo_de_processo":   {"critico": 2.5, "alto": 2.0, "medio": 1.2, "baixo": 0.5},
    "turret":               {"critico": 2.5, "alto": 1.8, "medio": 1.0, "baixo": 0.5},
    "topo_das_torres":      {"critico": 2.2, "alto": 1.8, "medio": 1.0, "baixo": 0.6},
    "area_offloading":      {"critico": 2.0, "alto": 1.6, "medio": 1.1, "baixo": 0.6},
    "moonpool":             {"critico": 1.8, "alto": 1.5, "medio": 1.1, "baixo": 0.7},
    "praça_de_máquinas":    {"critico": 1.5, "alto": 1.4, "medio": 1.2, "baixo": 0.8},
    "convés_principal":     {"critico": 1.2, "alto": 1.2, "medio": 1.2, "baixo": 1.0},
    "cais_de_barcos":       {"critico": 1.0, "alto": 1.2, "medio": 1.3, "baixo": 1.0},
    "área_de_utilidades":   {"critico": 0.8, "alto": 1.0, "medio": 1.3, "baixo": 1.4},
    "alojamento":           {"critico": 0.4, "alto": 0.6, "medio": 1.2, "baixo": 2.0},
    "sala_de_controle":     {"critico": 0.3, "alto": 0.5, "medio": 1.1, "baixo": 2.2},
}

# ── Frente 2: equipment class weights per risk class ─────────────────────────
# High-energy equipment overrepresented in critico/alto.
_EQUIP_CLASS_MULTIPLIERS: dict[str, dict[str, float]] = {
    "elevacao":          {"critico": 2.5, "alto": 2.0, "medio": 1.2, "baixo": 0.5},
    "eletrico":          {"critico": 2.2, "alto": 1.8, "medio": 1.2, "baixo": 0.6},
    "vaso_pressao":      {"critico": 2.0, "alto": 1.8, "medio": 1.3, "baixo": 0.6},
    "rotativo":          {"critico": 1.5, "alto": 1.5, "medio": 1.3, "baixo": 0.8},
    "mergulho_submarino":{"critico": 1.8, "alto": 1.5, "medio": 1.0, "baixo": 0.6},
    "instrumentacao":    {"critico": 1.0, "alto": 1.2, "medio": 1.3, "baixo": 1.0},
    "estrutural":        {"critico": 1.2, "alto": 1.3, "medio": 1.2, "baixo": 0.9},
    "protecao_incendio": {"critico": 0.8, "alto": 0.9, "medio": 1.2, "baixo": 1.8},
}

# ── Frente 3: risk factor distributions per class ─────────────────────────────
# Maps class → {factor_category → probability weight}.
# critico/alto overweight accident/chemical; baixo/medio overweight ergonomic/physical.
_FATOR_CLASS_WEIGHTS: dict[str, dict[str, float]] = {
    "critico": {"acidente": 0.50, "quimico": 0.25, "fisico": 0.15, "ergonomico": 0.05, "biologico": 0.05},
    "alto":    {"acidente": 0.40, "quimico": 0.20, "fisico": 0.25, "ergonomico": 0.10, "biologico": 0.05},
    "medio":   {"acidente": 0.20, "quimico": 0.15, "fisico": 0.30, "ergonomico": 0.30, "biologico": 0.05},
    "baixo":   {"acidente": 0.10, "quimico": 0.10, "fisico": 0.30, "ergonomico": 0.40, "biologico": 0.10},
}


def _weighted_area(areas_cfg: list, risk_class: str, rng: random.Random) -> str:
    ids = [a["id"] for a in areas_cfg]
    base_weights = [a["risk_weight"] for a in areas_cfg]
    weights = [
        bw * _AREA_CLASS_MULTIPLIERS.get(aid, {}).get(risk_class, 1.0)
        for aid, bw in zip(ids, base_weights)
    ]
    return rng.choices(ids, weights=weights, k=1)[0]


def _weighted_equip(equipment_map: dict, risk_class: str, rng: random.Random) -> tuple[str, str]:
    equip_classes = list(equipment_map.keys())
    weights = [
        _EQUIP_CLASS_MULTIPLIERS.get(ec, {}).get(risk_class, 1.0)
        for ec in equip_classes
    ]
    equip_cls = rng.choices(equip_classes, weights=weights, k=1)[0]
    equip_sub = rng.choice(equipment_map[equip_cls])
    return equip_cls, equip_sub


def _weighted_fator(risk_factors_cfg: dict, risk_class: str, rng: random.Random) -> str:
    category_weights = _FATOR_CLASS_WEIGHTS[risk_class]
    categories = list(risk_factors_cfg.keys())
    cat_weights = [category_weights.get(c, 0.1) for c in categories]
    chosen_cat = rng.choices(categories, weights=cat_weights, k=1)[0]
    return rng.choice(risk_factors_cfg[chosen_cat])


# ── Main generator ─────────────────────────────────────────────────────────────

def generate(data_cfg: dict, labels_cfg: dict, rng: random.Random) -> pd.DataFrame:
    gen = data_cfg["generation"]
    total = gen["total_records"]
    dist = gen["class_distribution"]
    noise_cfg = gen["noise"]
    text_cfg = gen["text"]

    ordered_classes = [c["id"] for c in labels_cfg["risk_classes"]]
    chemicals = [c["id"] for c in labels_cfg["chemical_products"]]
    incident_types = labels_cfg["incident_types"]

    equipment_map: dict[str, list[str]] = {
        cls_id: info["subclasses"]
        for cls_id, info in labels_cfg["equipment"].items()
    }

    base_date = date(2022, 1, 1)
    date_range = (date(2025, 12, 31) - base_date).days

    records = []

    for _ in range(total):
        risk_class = sample_class(dist, rng)
        # Frente 2: area and equipment sampled with class-specific weights
        area_id = _weighted_area(labels_cfg["fpso_areas"], risk_class, rng)
        equip_cls, equip_sub = _weighted_equip(equipment_map, risk_class, rng)
        chemical_id = rng.choice(chemicals)
        incident_type = rng.choice(incident_types)
        turno = rng.choice(["A", "B", "C"])
        # Frente 3: risk factor sampled from class-specific category distribution
        fator = _weighted_fator(labels_cfg["risk_factors"], risk_class, rng)
        ocorrencia_date = base_date + timedelta(days=rng.randint(0, date_range))

        hora, minuto = sample_time_of_day(turno, risk_class, rng)
        ocorrencia_dt = datetime(
            ocorrencia_date.year, ocorrencia_date.month, ocorrencia_date.day,
            hora, minuto, rng.randint(0, 59),
            tzinfo=_TZ_BRASIL,
        )

        include_chem = rng.random() < text_cfg["chemical_mention_prob"]
        include_norm = rng.random() < text_cfg["norm_mention_prob"]
        include_time = rng.random() < text_cfg["time_mention_prob"]
        include_weather = rng.random() < 0.25
        include_worker_state = rng.random() < 0.20

        relato = build_narrative(
            risk_class=risk_class,
            area_id=area_id,
            equip_subclass=equip_sub,
            chemical_id=chemical_id,
            fator=fator,
            turno=turno,
            include_chemical=include_chem,
            include_norm=include_norm,
            include_time=include_time,
            include_weather=include_weather,
            include_worker_state=include_worker_state,
            rng=rng,
            location_prep=LOCATION_PREP,
            equipment_context=EQUIPMENT_CONTEXT,
            chemical_desc={k: rng.choice(v) if isinstance(v, list) else v for k, v in CHEMICAL_DESC.items()},
            fator_desc={k: rng.choice(v) if isinstance(v, list) else v for k, v in FATOR_RISCO_DESC.items()},
            turno_desc=TURNO_DESC,
        )

        is_unannotated = rng.random() < noise_cfg["unannotated_rate"]
        is_mislabeled = False
        is_ambiguous = False
        label = risk_class

        if not is_unannotated:
            if rng.random() < noise_cfg["mislabeled_rate"]:
                wrong_pool = [c for c in ordered_classes if c != risk_class]
                label = rng.choice(wrong_pool)
                is_mislabeled = True
            elif rng.random() < noise_cfg["ambiguous_rate"]:
                label = adjacent_class(risk_class, ordered_classes, rng)
                is_ambiguous = True

        records.append({
            "id": str(uuid.uuid4()),
            "data_ocorrencia": ocorrencia_date,
            "data_hora_ocorrencia": ocorrencia_dt,
            "turno": turno,
            "area_fpso": area_id,
            "equipamento_classe": equip_cls,
            "equipamento_subclasse": equip_sub,
            "produto_quimico": chemical_id,
            "tipo_ocorrencia": incident_type,
            "fator_risco": fator,
            "relato": relato,
            "classe_risco": None if is_unannotated else label,
            "anotado": not is_unannotated,
            "ruido": is_mislabeled,
            "ambiguo": is_ambiguous,
        })

    return pd.DataFrame(records)


# ── Split ──────────────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame, data_cfg: dict, root: Path) -> None:
    from sklearn.model_selection import train_test_split

    parquet_opts = {
        "engine": data_cfg["parquet"]["engine"],
        "compression": data_cfg["parquet"]["compression"],
        "index": data_cfg["parquet"]["index"],
    }

    raw_path = root / data_cfg["paths"]["raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, **parquet_opts)
    print(f"[OK] Raw dataset saved     : {raw_path}  ({len(df):,} records)")

    annotated = df[df["anotado"]].copy()
    unannotated = df[~df["anotado"]].copy()

    split_cfg = data_cfg["split"]
    train_df, test_df = train_test_split(
        annotated,
        test_size=split_cfg["test_size"],
        stratify=annotated[split_cfg["stratify_by"]],
        random_state=42,
    )

    processed_root = root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    train_path = root / data_cfg["paths"]["train"]
    test_path = root / data_cfg["paths"]["test"]
    unann_path = root / data_cfg["paths"]["unannotated"]

    train_df.to_parquet(train_path, **parquet_opts)
    test_df.to_parquet(test_path, **parquet_opts)
    unannotated.to_parquet(unann_path, **parquet_opts)

    print(f"[OK] Train set saved       : {train_path}  ({len(train_df):,} records)")
    print(f"[OK] Test set saved        : {test_path}  ({len(test_df):,} records)")
    print(f"[OK] Unannotated saved     : {unann_path}  ({len(unannotated):,} records)")

    print("\n--- Class distribution (train) ---")
    print(train_df["classe_risco"].value_counts(normalize=True).round(3).to_string())
    print("\n--- Noise summary (full dataset) ---")
    print(f"  Mislabeled : {df['ruido'].sum():>6,}  ({df['ruido'].mean():.1%})")
    print(f"  Ambiguous  : {df['ambiguo'].sum():>6,}  ({df['ambiguo'].mean():.1%})")
    print(f"  Unannotated: {(~df['anotado']).sum():>6,}  ({(~df['anotado']).mean():.1%})")

    print("\n--- Narrative length stats (relato) ---")
    lengths = df["relato"].str.split().str.len()
    print(f"  Min words  : {lengths.min()}")
    print(f"  Max words  : {lengths.max()}")
    print(f"  Mean words : {lengths.mean():.1f}")
    print(f"  Unique     : {df['relato'].nunique():,} / {len(df):,}  ({df['relato'].nunique()/len(df):.1%})")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FPSO safety records")
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to data config (relative to project root)",
    )
    parser.add_argument(
        "--labels-config",
        default="configs/labels.yaml",
        help="Path to labels config (relative to project root)",
    )
    args = parser.parse_args()

    root = resolve_project_root()
    data_cfg = load_config(root / args.data_config)
    labels_cfg = load_config(root / args.labels_config)

    rng = random.Random(data_cfg["generation"]["random_seed"])

    print(f"Generating {data_cfg['generation']['total_records']:,} records...")
    df = generate(data_cfg, labels_cfg, rng)
    split_and_save(df, data_cfg, root)
    print("\nDone.")


if __name__ == "__main__":
    main()
