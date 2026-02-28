# MEVA Representative Slots

Analysis across 381 indexed slots (220 with Kitware activity annotations).  
Dataset: 37 activity types, 4 sites, 7 dates (Mar 5–15 2018), hours 10–17.

---

## Compact Set — 8 Slots (Maximum Per-Site Richness)

Covers all 4 sites, all 3 time-of-day periods, 3 different dates.

| Slot | Site | Time of Day | Date | Cams | Act Types | Annotations |
|------|------|-------------|------|-----:|----------:|------------:|
| `2018-03-15.15-10.school` | school | afternoon | Mar 15 | 13 | **30** | 351 |
| `2018-03-07.11-00.school` | school | morning | Mar 7 | 11 | 29 | 575 |
| `2018-03-15.15-45.bus` | bus | afternoon | Mar 15 | 6 | **31** | 148 |
| `2018-03-07.17-30.bus` | bus | evening | Mar 7 | 6 | 28 | 290 |
| `2018-03-15.15-50.hospital` | hospital | afternoon | Mar 15 | 3 | **27** | 53 |
| `2018-03-07.17-05.hospital` | hospital | evening | Mar 7 | 3 | 26 | 140 |
| `2018-03-07.11-05.admin` | admin | morning | Mar 7 | 2 | 5 | 17 |
| `2018-03-11.16-20.admin` | admin | afternoon | Mar 11 | 2 | 5 | 22 |

---

## Extended Set — 12 Slots (All 7 Recording Dates)

Add these 4 to the compact set to cover every date in the dataset:

| Slot | Site | Date | Cams | Act Types | Notes |
|------|------|------|-----:|----------:|-------|
| `2018-03-09.10-15.school` | school | Mar 9 | 13 | 27 | Densest annotations (910) |
| `2018-03-11.16-25.school` | school | Mar 11 | 13 | 29 | High diversity, 457 anns |
| `2018-03-05.13-20.school` | school | Mar 5 | 11 | 25 | Only high-quality Mar 5 slot |
| `2018-03-13.17-10.school` | school | Mar 13 | 13 | 23 | Evening, Mar 13 |

---

## Top Slots Per Site (Full Ranked Lists)

### School (13 cameras typical)

| Slot | Cams | Act Types | Annotations | Activity Types |
|------|-----:|----------:|------------:|----------------|
| `2018-03-15.15-10.school` | 13 | 30 | 351 | hand_interacts_with_person, person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_interacts_with_laptop, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_rides_bicycle, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_drops_off_person, vehicle_makes_u_turn, vehicle_picks_up_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-15.15-15.school` | 13 | 30 | 302 | person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_interacts_with_laptop, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_purchases, person_puts_down_object, person_rides_bicycle, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_makes_u_turn, vehicle_picks_up_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-11.16-25.school` | 13 | 29 | 457 | empty_37, hand_interacts_with_person, person_carries_heavy_object, person_closes_facility_door, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_reads_document, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_picks_up_person, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-07.11-00.school` | 11 | 29 | 575 | hand_interacts_with_person, person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_enters_scene_through_structure, person_exits_scene_through_structure, person_exits_vehicle, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_purchases, person_puts_down_object, person_reads_document, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_drops_off_person, vehicle_makes_u_turn, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-15.15-35.school` | 11 | 28 | 306 | person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_interacts_with_laptop, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_purchases, person_puts_down_object, person_rides_bicycle, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_drops_off_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-09.10-15.school` | 13 | 27 | **910** | hand_interacts_with_person, person_carries_heavy_object, person_closes_facility_door, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_drops_off_person, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-11.17-25.school` | 13 | 27 | 951 | hand_interacts_with_person, person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_picks_up_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |

### Bus (6 cameras typical)

| Slot | Cams | Act Types | Annotations | Activity Types |
|------|-----:|----------:|------------:|----------------|
| `2018-03-15.15-45.bus` | 6 | 31 | 148 | hand_interacts_with_person, person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_rides_bicycle, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_drops_off_person, vehicle_makes_u_turn, vehicle_picks_up_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-07.17-30.bus` | 6 | 28 | 290 | hand_interacts_with_person, person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_reads_document, person_sits_down, person_stands_up, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_drops_off_person, vehicle_picks_up_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-07.11-05.bus` | 6 | 26 | 235 | person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_reads_document, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-07.17-00.bus` | 6 | 26 | 372 | person_carries_heavy_object, person_closes_vehicle_door, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_loads_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_purchases, person_puts_down_object, person_reads_document, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |

### Hospital (3 cameras typical)

| Slot | Cams | Act Types | Annotations | Activity Types |
|------|-----:|----------:|------------:|----------------|
| `2018-03-15.15-50.hospital` | 3 | 27 | 53 | hand_interacts_with_person, person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_drops_off_person, vehicle_makes_u_turn, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-07.17-05.hospital` | 3 | 26 | 140 | person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_drops_off_person, vehicle_makes_u_turn, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-15.15-45.hospital` | 3 | 25 | 183 | hand_interacts_with_person, person_carries_heavy_object, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_enters_vehicle, person_exits_scene_through_structure, person_exits_vehicle, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_rides_bicycle, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, vehicle_drops_off_person, vehicle_makes_u_turn, vehicle_picks_up_person, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |
| `2018-03-07.11-00.hospital` | 3 | 24 | 112 | person_carries_heavy_object, person_closes_trunk, person_closes_vehicle_door, person_embraces_person, person_enters_scene_through_structure, person_exits_scene_through_structure, person_exits_vehicle, person_opens_facility_door, person_opens_trunk, person_opens_vehicle_door, person_picks_up_object, person_puts_down_object, person_sits_down, person_stands_up, person_talks_on_phone, person_talks_to_person, person_texts_on_phone, person_transfers_object, person_unloads_vehicle, vehicle_reverses, vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right |

### Admin (2 cameras — inherently sparse)

| Slot | Cams | Act Types | Annotations | Activity Types |
|------|-----:|----------:|------------:|----------------|
| `2018-03-07.11-05.admin` | 2 | 5 | 17 | person_closes_facility_door, person_enters_scene_through_structure, person_exits_scene_through_structure, person_opens_facility_door, person_talks_to_person |
| `2018-03-11.16-20.admin` | 2 | 5 | 22 | person_enters_scene_through_structure, person_exits_scene_through_structure, person_opens_facility_door, person_talks_on_phone, person_talks_to_person |
| `2018-03-09.10-10.admin` | 2 | 5 | 40 | person_closes_facility_door, person_enters_scene_through_structure, person_exits_scene_through_structure, person_opens_facility_door, person_talks_to_person |

---

## Global Activity Type Distribution (37 types)

Frequency = number of slots containing that activity type.

| Activity | Slots |
|----------|------:|
| person_talks_to_person | 179 |
| person_enters_scene_through_structure | 153 |
| person_exits_scene_through_structure | 151 |
| vehicle_turns_right | 142 |
| vehicle_turns_left | 137 |
| person_opens_facility_door | 134 |
| vehicle_stops | 133 |
| vehicle_starts | 130 |
| person_texts_on_phone | 119 |
| person_closes_vehicle_door | 118 |
| person_opens_vehicle_door | 118 |
| person_picks_up_object | 111 |
| person_puts_down_object | 100 |
| person_talks_on_phone | 99 |
| person_exits_vehicle | 89 |
| person_stands_up | 87 |
| person_transfers_object | 84 |
| person_enters_vehicle | 82 |
| vehicle_reverses | 81 |
| person_sits_down | 79 |
| hand_interacts_with_person | 78 |
| person_carries_heavy_object | 72 |
| person_embraces_person | 55 |
| vehicle_makes_u_turn | 47 |
| person_closes_trunk | 46 |
| person_opens_trunk | 45 |
| vehicle_drops_off_person | 40 |
| person_reads_document | 31 |
| person_unloads_vehicle | 28 |
| vehicle_picks_up_person | 27 |
| person_loads_vehicle | 20 |
| person_purchases | 17 |
| person_rides_bicycle | 17 |
| person_closes_facility_door | 11 |
| person_interacts_with_laptop | 6 |
| person_steals_object | 1 |

### Rare Activity Notes
- `person_steals_object` — only 1 slot in the entire dataset
- `person_interacts_with_laptop` — 6 slots, mostly school Mar 15 (`15-10`, `15-15`, `15-35`)
- `person_purchases` — 17 slots, concentrated in school/bus Mar 7

---

## Key Notes

- **Admin site is structurally sparse**: only 2 cameras, max 5 activity types observed — not a pipeline limitation
- **`2018-03-09.10-15.school`** is the single densest slot (910 annotations, 27 types, 13 cameras) — best for stress-testing
- **`2018-03-15.15-45.bus`** has the highest global activity type count (31/37 types)
- **Development slot recommendation**: `2018-03-15.15-10.school` (30 types, 13 cams, afternoon, Mar 15)
- **Cross-site validation recommendation**: `2018-03-07.17-30.bus` (28 types, 6 cams, evening, Mar 7)
- 220/381 slots have Kitware activity annotations; the remaining 161 have only camera coverage in the index

---

*Generated: 2026-02-27 | Source: Kitware KPF YAML annotations + slot_index.json (381 slots)*
