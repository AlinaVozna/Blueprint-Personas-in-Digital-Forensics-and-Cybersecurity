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

persona(lena). intention(lena, malicious).
technique(lena, exploit, high_severity).
technique(lena, scan, low_severity).
technique(lena, phishing, high_severity).
technique(lena, lateral_movement, high_severity).

persona(bob). intention(bob, benevolent). 
technique(bob, login, neutral).
technique(bob, powershell, high_severity).

persona(alice). intention(alice, benevolent).
technique(alice, revoke_access, neutral).

persona(charlie). intention(charlie, benevolent).
technique(charlie, db_access, neutral).

persona(diana). intention(diana, benevolent).
technique(diana, containment, neutral).
technique(diana, escalate, neutral).

persona(eve). intention(eve, benevolent).
technique(eve, data_exfil, high_severity).

% --- 3. DYNAMIC OBSERVATIONS (Input from Python) ---
% observed(Agent, Action). 

% --- 4. ROT: TRUST INFERENCE LOGIC ---

base_trust(Agent, low)    :- intention(Agent, malicious).
base_trust(Agent, high)   :- intention(Agent, benevolent).
% Safe variable fix
base_trust(Agent, medium) :- 
    persona(Agent), 
    not intention(Agent, malicious), 
    not intention(Agent, benevolent).

derived_trust(Agent, very_low) :- 
    observed(Agent, Action), technique(Agent, Action, high_severity), base_trust(Agent, low).

derived_trust(Agent, medium) :- 
    observed(Agent, Action), technique(Agent, Action, high_severity), base_trust(Agent, high).

derived_trust(Agent, low) :- 
    observed(Agent, Action), technique(Agent, Action, low_severity), base_trust(Agent, low).

derived_trust(Agent, high) :- 
    observed(Agent, Action), technique(Agent, Action, neutral), base_trust(Agent, high).

derived_trust(charlie, medium) :- 
    observed(charlie, db_access). 

% Fallback
derived_trust(Agent, Level) :- 
    base_trust(Agent, Level), not exception_triggered(Agent).
    
exception_triggered(Agent) :- derived_trust(Agent, X), base_trust(Agent, Y), X != Y.

% --- 5. OUTPUT HELPER ---
computed_trust_val(Agent, Val) :- 
    derived_trust(Agent, Label), trust_level(Label, Val).

#show computed_trust_val/2.
"""

# ==============================================================================
# 2. COGNITIVE ENGINE (PYTHON LAYER)
# ==============================================================================
class CognitiveEngine:
    def solve_scenario(self, agent_name, observed_action, required_trust_val):
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
        
        if not trust_found: trust_val = 2 

        display_trust = trust_val / 5.0 
        required_float = required_trust_val / 5.0
        
        decision = ""
        if trust_val >= required_trust_val:
            decision = "ALLOW"
        elif trust_val <= 1: 
            decision = "BLOCK"
        else:
            decision = "DELEGATE" # Gray Zone

        return display_trust, decision

# ==============================================================================
# 3. EXPERIMENTS
# ==============================================================================
def run_phase1_pivot():
    print("="*70)
    print("PHASE 1: Multi-Step Attack (Pivot)")
    print("="*70)
    engine = CognitiveEngine()
    steps = [
        {"s": "T1", "ag": "lena", "act": "scan", "req": 2},
        {"s": "T2", "ag": "lena", "act": "exploit", "req": 4},
        {"s": "T3", "ag": "bob", "act": "login", "req": 4},
        {"s": "T4", "ag": "bob", "act": "powershell", "req": 4}
    ]
    for x in steps:
        val, dec = engine.solve_scenario(x['ag'], x['act'], x['req'])
        print(f"[{x['s']}] {x['ag']:<6} | {x['act']:<10} | Trust: {val:.2f} | Dec: {dec}")

def run_phase2_complex():
    print("\n" + "="*70)
    print("PHASE 2: Complex Multi-Agent Scenario (New in Revision)")
    print("="*70)
    engine = CognitiveEngine()
    
    steps = [
        {"s": "T1", "ag": "lena", "act": "scan", "req": 2},
        {"s": "T2", "ag": "lena", "act": "phishing", "req": 4},
        {"s": "T3", "ag": "charlie", "act": "db_access", "req": 4},
        {"s": "T4", "ag": "diana", "act": "containment", "req": 4},
        {"s": "T5", "ag": "eve", "act": "data_exfil", "req": 4},
        {"s": "T6", "ag": "alice", "act": "revoke_access", "req": 4},
        {"s": "T7", "ag": "lena", "act": "lateral_movement", "req": 4},
        {"s": "T8", "ag": "diana", "act": "escalate", "req": 4}
    ]
    
    print(f"{'Step':<4} | {'Agent':<8} | {'Action':<15} | {'Trust':<6} | {'Decision'}")
    print("-" * 60)
    
    for x in steps:
        val, dec = engine.solve_scenario(x['ag'], x['act'], x['req'])
        
        c_dec = f"\033[92m{dec}\033[0m" if dec=="ALLOW" else f"\033[91m{dec}\033[0m" if dec=="BLOCK" else f"\033[93m{dec}\033[0m"
        print(f"{x['s']:<4} | {x['ag']:<8} | {x['act']:<15} | {val:.2f}   | {c_dec}")

def run_phase3_scalability():
    print("\n" + "="*70)
    print("PHASE 3: Scalability & Breakdown (New Breakdown Metrics)")
    print("="*70)
    
    kb_sizes = [100, 1000, 5000, 10000, 20000]
    
    print(f"{'KB Size':<10} | {'Ground (ms)':<12} | {'Solve (ms)':<10} | {'Total (ms)':<10}")
    print("-" * 55)

    for size in kb_sizes:
        noise = "".join([f'log(id_{i}, "act", "ip"). ' for i in range(size)])
        
        program = asp_program_base + "\n" + noise + """
        persona(stress). intention(stress, benevolent). technique(stress, login, neutral).
        observed(stress, login).
        """
        
        # 1. Start Timer
        t0 = time.time()
        
        ctl = clingo.Control(["0", "--warn=none"])
        ctl.add("base", [], program)
        
        # 2. Measure Grounding
        t1 = time.time()
        ctl.ground([("base", [])])
        t2 = time.time()
        
        # 3. Measure Solving
        ctl.solve()
        t3 = time.time()
        
        ground_time = (t2 - t1) * 1000
        solve_time = (t3 - t2) * 1000
        total_time = (t3 - t0) * 1000 # Include overhead di add e setup
        
        print(f"{size:<10} | {ground_time:<12.2f} | {solve_time:<10.2f} | {total_time:<10.2f}")

if __name__ == "__main__":
    run_phase1_pivot()
    run_phase2_complex()
    run_phase3_scalability()