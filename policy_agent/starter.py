"""Eviction controller. Return None to retain the selected starting policy."""
class Policy:
    def __init__(self):
        self.blocks = {}

    def on_event(self, event):
        kind = event['kind']
        block_id = event.get('block_id')
        if kind == 'add':
            self.blocks[block_id] = dict(event)
        elif kind == 'remove':
            self.blocks.pop(block_id, None)
        elif kind == 'update' and block_id in self.blocks:
            self.blocks[block_id].update(event)

    def evict(self, now):
        # Return an eligible block ID to override the starting policy.
        return None
