% BLUEPRINT PERSONAS

% Definition of persona "Lena"
persona("Lena").
age("Lena", 28).
affiliation("Lena", "APT-linked nation-state").
specialization("Lena", "targeting government and financial institutions").

% Lena's objectives
objective("Lena", infiltrate_systems).
objective("Lena", exfiltrate_data).
objective("Lena", maintain_access).

% Techniques used by Lena
technique("Lena", zero_day_exploits).
technique("Lena", phishing_attacks).
technique("Lena", command_and_control).
technique("Lena", data_exfiltration).

% Tools used by Lena
tool("Lena", tor_networks).
tool("Lena", encrypted_c2_servers).
tool("Lena", malware_apt).

% BEHAVIORAL RULES
uses_technique(Persona, Technique, Objective) :-
    persona(Persona),
    technique(Persona, Technique),
    objective(Persona, Objective),
    requires_technique(Objective, Technique).

uses_tool(Persona, Tool, Technique) :-
    persona(Persona),
    tool(Persona, Tool),
    technique(Persona, Technique),
    supports_tool(Technique, Tool).

% DEPENDENCY MAPPING
% Which techniques are required for each objective
requires_technique(infiltrate_systems, zero_day_exploits).
requires_technique(infiltrate_systems, phishing_attacks).
requires_technique(maintain_access, command_and_control).
requires_technique(exfiltrate_data, data_exfiltration).

% Which tools support which techniques
supports_tool(zero_day_exploits, malware_apt).
supports_tool(phishing_attacks, malware_apt).
supports_tool(command_and_control, tor_networks).
supports_tool(command_and_control, encrypted_c2_servers).
supports_tool(data_exfiltration, encrypted_c2_servers).

% MAIN ATTACK RULE
attack(Persona, Objective) :-
    persona(Persona),
    objective(Persona, Objective),
    uses_technique(Persona, _, Objective),
    not missing_tool(Persona, Objective).

% Missing tool for an objective
missing_tool(Persona, Objective) :-
    requires_technique(Objective, Technique),
    supports_tool(Technique, Tool),
    persona(Persona),
    not tool(Persona, Tool).

% ADVANCED ATTACK SIMULATION
complete_attack(Persona, Objective, Technique, Tool) :-
    attack(Persona, Objective),
    uses_technique(Persona, Technique, Objective),
    uses_tool(Persona, Tool, Technique).

% ATTACK CONTEXT
government_target("Department of Defense").
financial_target("Major Bank XYZ").
vulnerable(System) :- government_target(System).
vulnerable(System) :- financial_target(System).

attack_target(Persona, System) :-
    attack(Persona, infiltrate_systems),
    vulnerable(System).

% QUERY OUTPUT
#show attack/2.
#show complete_attack/4.
#show attack_target/2.