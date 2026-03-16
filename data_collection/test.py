
import json
from clashapi import get_player
player = get_player("%238928P288L")
print(json.dumps(player, indent=2)[:2000])
print([k for k in player.keys()])

if "currentDeck" in player:
    print(json.dumps(player["currentDeck"][0], indent=2))