"""Opt-in compact feedback using the unchanged explicit-target actor transport."""
from online_bound_actor import BoundActor
from online_compact_geometry import compact_prompt


class CompactBoundActor(BoundActor):
    def request(self, prompt, images, *, slow, advance, deadline):
        transformed, encoded = compact_prompt(prompt, explain=not getattr(self, '_compact_contract_sent', False))
        result = super().request(transformed, images, slow=slow, advance=advance, deadline=deadline)
        if encoded:
            self._compact_contract_sent = True
        return result
