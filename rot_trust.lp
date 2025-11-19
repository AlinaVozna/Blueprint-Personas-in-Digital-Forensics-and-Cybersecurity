% REFERENCE ONTOLOGY OF TRUST (ROT) 

% Trust entities
trustor("SOC_System").
trustee("Lena").

% Trust factors
capability("Lena", high) :- 
    technique("Lena", zero_day_exploits),
    technique("Lena", phishing_attacks).

vulnerability("Lena", low) :-
    tool("Lena", tor_networks),
    tool("Lena", encrypted_c2_servers).

intention("Lena", malicious).

% Trust degree calculation
trust_degree(Trustor, Trustee, low) :-
    trustor(Trustor),
    trustee(Trustee),
    intention(Trustee, malicious).

trust_degree(Trustor, Trustee, very_low) :-
    trust_degree(Trustor, Trustee, low),
    capability(Trustee, high),
    vulnerability(Trustee, low).

% Security decisions based on trust
block_access(Persona) :-
    trust_degree(_, Persona, very_low).

% ADAPTIVE SECURITY RULES
attack_detected(Persona, Objective) :-
    attack(Persona, Objective),
    not block_access(Persona).

attack_blocked(Persona, Objective) :-
    attack(Persona, Objective),
    block_access(Persona).

% TRUST OUTPUT QUERIES
#show trust_degree/3.
#show block_access/1.
#show attack_detected/2.
#show attack_blocked/2.