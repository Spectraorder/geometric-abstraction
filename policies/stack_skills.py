"""Stack small_red_block on blue_block using the verified skill layer."""

from aspire_so100.skills import Skills


def run(api) -> None:
    sk = Skills(api)
    sk.pick_and_place("small_red_block", "blue_block", mode="contact")
    sk.go_clear()
