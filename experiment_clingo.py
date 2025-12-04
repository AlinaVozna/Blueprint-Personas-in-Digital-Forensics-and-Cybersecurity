import clingo
import time

# ==============================================================================
# 1. ASP KNOWLEDGE BASE (PERSONAS + ROT)
# ==============================================================================
asp_program_base = """
% --- 1. DOMAIN & TRUST LEVELS ---
trust_level(very_low, 1). trust_level(low, 2). 
trust_level(medium, 3). trust_level(high, 4). trust_level(very_high, 5).

% --- 2. BLUEPRINT PERSONAS (STATIC PROFILES) ---
persona(lena).
intention(lena, malicious).
technique(lena, exploit, high_severity).
technique(lena, scan, low_severity).

persona(bob).
intention(bob, benevolent). 
technique(bob, login, neutral).
technique(bob, powershell, high_severity).

% --- 3. DYNAMIC OBSERVATIONS (Input from Python) ---
% observed(Agent, Action).  <-- Iniettato da Python

% --- 4. ROT: TRUST INFERENCE LOGIC ---

% A. Calcolo della Fiducia Base (dalle intenzioni della Persona)
base_trust(Agent, low)    :- intention(Agent, malicious).
base_trust(Agent, high)   :- intention(Agent, benevolent).

% *** CORREZIONE QUI SOTTO ***
% Aggiunto 'persona(Agent)' per rendere la variabile sicura (Safe Variable)
base_trust(Agent, medium) :- 
    persona(Agent), 
    not intention(Agent, malicious), 
    not intention(Agent, benevolent).

% B. Regole di Adattamento basate sull'Evento Osservato

% CASO 1: L'azione osservata corrisponde a una tecnica GRAVE della persona
derived_trust(Agent, very_low) :- 
    observed(Agent, Action),
    technique(Agent, Action, high_severity),
    base_trust(Agent, low).

derived_trust(Agent, medium) :- 
    observed(Agent, Action),
    technique(Agent, Action, high_severity),
    base_trust(Agent, high).

% CASO 2: Azioni a bassa severità
derived_trust(Agent, low) :- 
    observed(Agent, Action),
    technique(Agent, Action, low_severity),
    base_trust(Agent, low).

% CASO 3: Azioni neutre
derived_trust(Agent, high) :- 
    observed(Agent, Action),
    technique(Agent, Action, neutral),
    base_trust(Agent, high).

% Fallback: Se non c'è un match specifico, mantieni la fiducia base
% Anche qui aggiungiamo persona(Agent) per sicurezza, anche se base_trust lo rende safe
derived_trust(Agent, Level) :- 
    base_trust(Agent, Level),
    not exception_triggered(Agent).
    
exception_triggered(Agent) :- derived_trust(Agent, X), base_trust(Agent, Y), X != Y.

% --- 5. OUTPUT HELPER ---
computed_trust_val(Agent, Val) :- 
    derived_trust(Agent, Label), 
    trust_level(Label, Val).

#show computed_trust_val/2.
"""

# ==============================================================================
# 2. COGNITIVE ENGINE (PYTHON LAYER)
# ==============================================================================
class CognitiveEngine:
    def __init__(self):
        pass

    def solve_scenario(self, agent_name, observed_action, required_trust_val):
        """
        Inietta l'osservazione corrente nel programma ASP e decide.
        """
        # Costruiamo il programma dinamico per questo step
        step_program = asp_program_base + f'\nobserved({agent_name}, {observed_action}).\n'

        ctl = clingo.Control(["0", "--warn=none"])
        ctl.add("base", [], step_program)
        ctl.ground([("base", [])])

        trust_val = 0
        trust_found = False
        
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for atom in model.symbols(shown=True):
                    if atom.name == "computed_trust_val":
                        if str(atom.arguments[0]) == agent_name:
                            trust_val = int(str(atom.arguments[1]))
                            trust_found = True
        
        if not trust_found:
            trust_val = 2 # Default Low se l'agente non viene risolto

        # L-DINF Logic
        decision = ""
        reason = ""
        
        if trust_val >= required_trust_val:
            decision = "ALLOW"
            reason = f"Trust ({trust_val}) >= Required ({required_trust_val})"
        elif trust_val <= 1: 
            decision = "BLOCK"
            reason = f"Trust ({trust_val}) is CRITICAL (Malicious + High Sev)"
        else:
            decision = "DELEGATE"
            reason = f"Trust ({trust_val}) < Required ({required_trust_val}) -> Ambiguity Detected"

        return trust_val, decision, reason

# ==============================================================================
# 3. EXPERIMENTS
# ==============================================================================
def run_multi_step_campaign():
    print("="*70)
    print("PHASE 1: Multi-Step Attack Campaign (Context-Aware Trust)")
    print("="*70)

    engine = CognitiveEngine()
    
    scenario_steps = [
        {
            "step": "T1", "agent": "lena", "action": "scan", "req": 2,
            "desc": "Reconnaissance: Lena scans public ports (Low Sev)."
        },
        {
            "step": "T2", "agent": "lena", "action": "exploit", "req": 4,
            "desc": "Attack: Lena uses Zero-Day Exploit (High Sev)."
        },
        {
            "step": "T3", "agent": "bob", "action": "login", "req": 4,
            "desc": "Normal Ops: Bob logs into DB (Neutral)."
        },
        {
            "step": "T4", "agent": "bob", "action": "powershell", "req": 4,
            "desc": "Insider Threat? Bob runs encoded PowerShell (High Sev)."
        }
    ]

    for s in scenario_steps:
        print(f"\n--- [ {s['step']} ] {s['desc']} ---")
        t_val, decision, reason = engine.solve_scenario(s['agent'], s['action'], s['req'])
        
        color = "\033[92m" if decision == "ALLOW" else "\033[91m" if decision == "BLOCK" else "\033[93m"
        reset = "\033[0m"
        
        print(f"Agent Profile: {s['agent'].capitalize()}")
        print(f"Observed Act : {s['action']}")
        print(f"Trust Level  : {t_val} / 5")
        print(f"Decision     : {color}{decision}{reset}")
        print(f"Reasoning    : {reason}")

def run_performance_test():
    print("\n" + "="*70)
    print("PHASE 2: Scalability Stress Test")
    print("="*70)
    
    kb_sizes = [100, 1000, 5000, 10000]
    print(f"{'KB Size (Facts)':<20} | {'Time (seconds)':<15}")
    print("-" * 40)

    for size in kb_sizes:
        # Generiamo log fittizi
        noise = "".join([f'log(id_{i}, "user_login", "192.168.1.{i%255}").\n' for i in range(size)])
        
        start_time = time.time()
        
        # Aggiungiamo un agente fittizio 'stress_test' per il test
        # Definiamo anche una intenzione base per renderlo 'safe' nella regola base_trust
        program = asp_program_base + "\n" + noise + """
        persona(stress_test). 
        intention(stress_test, benevolent).
        technique(stress_test, login, neutral).
        observed(stress_test, login).
        """
        
        ctl = clingo.Control(["0", "--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        ctl.solve()
        
        end_time = time.time()
        print(f"{size:<20} | {(end_time - start_time):.4f}s")

if __name__ == "__main__":
    run_multi_step_campaign()
    run_performance_test()