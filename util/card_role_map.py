CARD_ROLE_MAP = {
    # — Troops — #
    # Win Conditions
    "Hog Rider": {"win_condition":1,"tank":0,"support":0,"spell":0,"building":0,"cycle":0},
    "Royal Giant": {"win_condition":1,"tank":1,"support":0,"spell":0,"building":0,"cycle":0},
    "Miner": {"win_condition":1,"tank":0,"support":0,"spell":0,"building":0,"cycle":0},
    "Balloon": {"win_condition":1,"tank":0,"support":0,"spell":0,"building":0,"cycle":0},
    "Golem": {"win_condition":1,"tank":1,"support":0,"spell":0,"building":0,"cycle":0},
    "Mortar": {"win_condition":1,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "X-Bow": {"win_condition":1,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},

    # Tanks/Mega Troops
    "Knight": {"win_condition":0,"tank":1,"support":0,"spell":0,"building":0,"cycle":0},
    "P.E.K.K.A": {"win_condition":0,"tank":1,"support":0,"spell":0,"building":0,"cycle":0},
    "Ice Golem": {"win_condition":0,"tank":1,"support":0,"spell":0,"building":0,"cycle":0},
    "Giant Skeleton": {"win_condition":0,"tank":1,"support":0,"spell":0,"building":0,"cycle":0},
    "Battle Healer": {"win_condition":0,"tank":1,"support":1,"spell":0,"building":0,"cycle":0},

    # Standard Support Troops
    "Archers": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Skeletons": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":0,"cycle":1},
    "Goblins": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":1},
    "Spear Goblins": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":1},
    "Minions": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Minion Horde": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Musketeer": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Baby Dragon": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Wizard": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Valkyrie": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Mega Minion": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Electro Wizard": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Princess": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Dark Prince": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Bandit": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Battle Ram": {"win_condition":1,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},

    # Spells
    "Fireball": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},
    "Zap": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":1},
    "The Log": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":1},
    "Arrows": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},
    "Poison": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},
    "Lightning": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},
    "Rocket": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},
    "Earthquake": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},
    "Freeze": {"win_condition":0,"tank":0,"support":0,"spell":1,"building":0,"cycle":0},

    # Buildings
    "Cannon": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "Tombstone": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "Inferno Tower": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "Bomb Tower": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "Goblin Hut": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "Barbarian Hut": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},
    "Elixir Collector": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":1,"cycle":0},

    # Cycle / Utility Cards
    "Ice Spirit": {"win_condition":0,"tank":0,"support":0,"spell":0,"building":0,"cycle":1},
    "Fire Spirits": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":1},
    "Skeleton Barrel": {"win_condition":0,"tank":0,"support":1,"spell":0,"building":0,"cycle":0},
    "Goblin Barrel": {"win_condition":1,"tank":0,"support":0,"spell":0,"building":0,"cycle":0},

    # (…extend with any newer card you encounter)
}