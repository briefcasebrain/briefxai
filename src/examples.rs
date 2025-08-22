use crate::types::{ConversationData, Message};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};

pub fn generate_realistic_conversations(count: usize, seed: Option<u64>) -> Vec<ConversationData> {
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let mut conversations = Vec::new();

    // Define realistic conversation templates
    let templates = vec![
        // Technical Support
        vec![
            ("user", "My laptop won't connect to WiFi after the latest Windows update. It was working fine yesterday."),
            ("assistant", "I understand how frustrating WiFi issues can be. Let's troubleshoot this step by step. First, can you tell me if you see your WiFi network in the list of available networks?"),
            ("user", "Yes, I can see it, but when I try to connect it says 'Can't connect to this network'"),
            ("assistant", "This is a common issue after Windows updates. Let's try a few solutions:\n\n1. First, let's reset your network adapter:\n   - Right-click the Start button and select 'Device Manager'\n   - Expand 'Network adapters'\n   - Right-click your WiFi adapter and select 'Disable device'\n   - Wait 10 seconds, then right-click again and select 'Enable device'\n\nDid this help?"),
            ("user", "No, still the same error. Should I try restarting?"),
            ("assistant", "Before restarting, let's try resetting the TCP/IP stack:\n\n1. Open Command Prompt as administrator\n2. Type these commands one by one:\n   - netsh winsock reset\n   - netsh int ip reset\n   - ipconfig /release\n   - ipconfig /renew\n   - ipconfig /flushdns\n\n3. Restart your computer after running these commands\n\nThis often resolves connectivity issues after updates."),
        ],

        // Creative Writing Help
        vec![
            ("user", "I'm writing a sci-fi story about a colony on Mars. How would daily life realistically look like for the colonists?"),
            ("assistant", "Great premise! For a realistic Mars colony, daily life would be shaped by several key constraints:\n\n**Environmental Challenges:**\n- Living in pressurized habitats with recycled air\n- Water would be precious - extracted from ice or recycled\n- No stepping outside without a spacesuit\n- Communication delays with Earth (4-24 minutes each way)\n\n**Daily Routines:**\n- Strict exercise regimens to combat muscle atrophy in lower gravity\n- Greenhouse duty - growing food would be critical\n- Constant maintenance of life support systems\n- Research and exploration missions\n\n**Social Dynamics:**\n- Small, tight-knit community with no privacy\n- Psychological challenges from isolation\n- Carefully scheduled video calls with Earth family\n- Recreation would be crucial for mental health\n\nWould you like me to elaborate on any of these aspects?"),
            ("user", "Yes, tell me more about the psychological challenges. How would they cope?"),
            ("assistant", "The psychological challenges would be intense:\n\n**Isolation Effects:**\n- Earth would just be a bright star in the sky\n- No possibility of evacuation in emergencies\n- Missing Earth's nature - no real sky, weather, or seasons\n- Delayed communication prevents real-time conversations with loved ones\n\n**Coping Mechanisms:**\n- Virtual reality environments simulating Earth locations\n- Structured social activities and celebrations\n- Personal greenhouse spaces for psychological comfort\n- Art and creative pursuits as outlets\n- Regular psychological counseling via AI or trained colonists\n- Meditation and mindfulness practices\n- Maintaining Earth-like day/night cycles with artificial lighting\n\n**Community Support:**\n- Buddy systems for mental health monitoring\n- Shared meals as social anchors\n- Collaborative projects to maintain purpose\n- Cultural events celebrating Earth traditions\n\nThe colonists would likely develop their own unique culture over time, distinct from Earth."),
        ],

        // Cooking Advice
        vec![
            ("user", "I'm trying to make homemade pasta for the first time. Any tips?"),
            ("assistant", "How exciting! Homemade pasta is so rewarding. Here are my essential tips for beginners:\n\n**Basic Recipe (serves 4):**\n- 2 cups all-purpose flour\n- 3 large eggs\n- 1/2 teaspoon salt\n- 1 tablespoon olive oil\n\n**Key Tips:**\n1. Make a well in the flour, crack eggs in the center\n2. Gradually incorporate flour into eggs with a fork\n3. Knead for 10 minutes until smooth and elastic\n4. Rest the dough for 30 minutes (crucial!)\n5. Roll thin - you should see your hand through it\n\nDo you have a pasta maker or will you be rolling by hand?"),
            ("user", "I have a rolling pin. Is that going to make it much harder?"),
            ("assistant", "Not at all! Rolling by hand is totally doable and how pasta was made for centuries. Here's how to make it easier:\n\n**Hand-Rolling Tips:**\n1. Work with 1/4 of your dough at a time (keep rest covered)\n2. Start from the center and roll outward\n3. Rotate the dough 90° frequently for even thickness\n4. Dust with flour if it sticks, but not too much\n5. Aim for 1-2mm thickness for most pasta\n\n**Pro tip:** Roll it around your rolling pin occasionally and unroll it - this helps stretch it thinner.\n\nFor your first time, I'd suggest making fettuccine - just roll, then fold the sheet loosely and cut into strips. Much easier than trying to make ravioli!"),
            ("user", "Great! How long should I cook fresh pasta?"),
            ("assistant", "Fresh pasta cooks MUCH faster than dried - only 2-4 minutes!\n\n**Timing Guide:**\n- Thin pasta (angel hair): 1-2 minutes\n- Fettuccine/tagliatelle: 2-3 minutes\n- Thicker pasta: 3-4 minutes\n\n**How to tell it's done:**\n- It will float to the surface\n- Taste test - should be tender but still have a slight bite\n- Fresh pasta won't get as 'al dente' as dried\n\n**Important:** Have everything else ready before you drop the pasta in - sauce warm, plates heated, cheese grated. Fresh pasta waits for no one!\n\nAlso, save a cup of pasta water before draining - its starch helps bind the sauce."),
        ],

        // Health & Fitness
        vec![
            ("user", "I want to start exercising but I haven't worked out in years. Where should I begin?"),
            ("assistant", "It's great that you're ready to start! The key is beginning gradually to avoid injury and burnout. Here's a beginner-friendly approach:\n\n**Week 1-2: Build the Habit**\n- Start with 10-15 minute walks daily\n- Focus on consistency over intensity\n- Pick a specific time each day\n\n**Week 3-4: Add Variety**\n- Increase walks to 20-30 minutes\n- Add 2 days of basic bodyweight exercises:\n  • 10 wall push-ups\n  • 10 chair squats\n  • 10 standing marches\n  • Hold a plank for 10 seconds\n\n**Important:**\n- Listen to your body\n- Rest days are crucial\n- Stay hydrated\n\nWhat's your main goal - general fitness, weight loss, or building strength?"),
            ("user", "Mainly weight loss and just feeling healthier overall"),
            ("assistant", "Perfect! For weight loss and overall health, here's what I recommend:\n\n**Exercise Plan:**\n- **Cardio (3-4x/week):** Walking, swimming, or cycling - start at 20 mins, build to 45\n- **Strength (2x/week):** Simple bodyweight exercises to maintain muscle while losing weight\n- **Flexibility (daily):** 5-10 minutes of stretching\n\n**Remember: Weight loss is 80% diet, 20% exercise**\n\n**Sustainable Changes:**\n- Track your steps (aim for 7,000-10,000 daily)\n- Take stairs when possible\n- Park farther away\n- Stand up every hour if you have a desk job\n\n**Diet Tips (without drastic changes):**\n- Drink water before each meal\n- Use smaller plates\n- Eat protein with every meal\n- Don't eliminate foods, just reduce portions\n\n**Most Important:** Aim to lose 1-2 pounds per week maximum. Faster isn't sustainable!\n\nWould you like help creating a specific weekly schedule?"),
        ],

        // Language Learning
        vec![
            ("user", "I'm trying to learn Spanish on my own. What's the most effective approach?"),
            ("assistant", "Self-learning Spanish can be very effective with the right approach! Here's a structured plan:\n\n**Foundation (Months 1-2):**\n- Learn 1000 most common words (20-30 per day)\n- Master present tense conjugations\n- Use apps like Anki for spaced repetition\n- 30 minutes daily minimum\n\n**Building Skills (Months 3-4):**\n- Start reading children's books\n- Watch Spanish shows with Spanish subtitles\n- Practice speaking to yourself daily\n- Learn past and future tenses\n\n**Immersion Techniques:**\n- Change your phone to Spanish\n- Follow Spanish social media accounts\n- Listen to Spanish podcasts during commutes\n- Find online conversation partners\n\nWhat's your current level, complete beginner?"),
            ("user", "Yes, complete beginner. The grammar seems really overwhelming"),
            ("assistant", "Don't worry! Spanish grammar is actually quite logical once you see the patterns. Let's break it down:\n\n**Start Simple - Don't Learn Everything at Once:**\n\n**Week 1-2: Just focus on:**\n- I am = Soy/Estoy\n- I have = Tengo\n- I want = Quiero\n- I need = Necesito\n- I like = Me gusta\n\nWith just these 5 verbs + nouns, you can say hundreds of things!\n\n**The 80/20 Rule:**\n- Present tense covers 50% of conversation\n- Past preterite adds another 30%\n- Everything else is just 20%\n\n**Grammar Hack:**\nInstead of memorizing rules, learn phrase patterns:\n- \"Me gusta...\" (I like...)\n- \"Puedo...\" (I can...)\n- \"Tengo que...\" (I have to...)\n\n**My #1 Tip:**\nDon't aim for perfection. Native speakers will understand \"Yo querer café\" even if it's grammatically wrong. Communication first, perfection later!\n\nWant me to teach you 10 phrases you can use immediately?"),
        ],

        // Career Advice
        vec![
            ("user", "I've been at my job for 5 years and feel stuck. Should I look for something new or try to grow where I am?"),
            ("assistant", "This is a common crossroads, and the answer depends on several factors. Let's explore both options:\n\n**Signs it's time to leave:**\n- No growth opportunities despite conversations with management\n- Your values no longer align with the company\n- Consistent stress affecting your health/personal life\n- You've stopped learning\n- Industry/company is declining\n\n**Signs you should stay and grow:**\n- You haven't had honest career conversations with your manager\n- There are unexplored lateral moves\n- Company is growing but you haven't positioned yourself\n- Good culture/benefits that would be hard to replace\n\n**Questions to ask yourself:**\n1. Have you explicitly asked for new challenges/promotion?\n2. What specific skills do you want to develop?\n3. Is it the role, company, or industry that's the problem?\n4. What would need to change for you to be excited about work again?\n\nWhat resonates most with your situation?"),
            ("user", "I haven't really talked to my manager about it. I guess I assumed they'd notice I'm ready for more"),
            ("assistant", "Ah, this is incredibly common! Most managers aren't mind readers, and they often assume no news is good news. Here's how to change that:\n\n**Prepare for the Conversation:**\n1. Document your achievements from the last 2 years\n2. Research roles you're interested in (internal postings, LinkedIn)\n3. Identify specific skills/responsibilities you want\n4. Know your market value\n\n**The Conversation Framework:**\n\"I've really enjoyed my 5 years here and I'm ready to take on more. I'd like to discuss my career path and what opportunities might be available.\"\n\n**Be Specific:**\n- \"I'm interested in leading projects\"\n- \"I'd like to mentor new team members\"\n- \"I want to learn [specific skill]\"\n\n**If They Say No Growth Available:**\nThen you have clarity and can job search guilt-free!\n\n**If They're Supportive:**\nCreate a 90-day plan together with concrete steps\n\n**Timeline:**\nSchedule this conversation within 2 weeks. Don't wait for performance review season.\n\nThe worst thing is wondering \"what if.\" Would you feel comfortable having this conversation?"),
        ],

        // Personal Finance
        vec![
            ("user", "I'm 28 and have no savings. How do I start building an emergency fund when I live paycheck to paycheck?"),
            ("assistant", "First, know that you're not alone - 60% of Americans are in the same situation. Let's build a realistic plan:\n\n**Start Micro - Week 1:**\n- Save just $1 this week. Yes, really.\n- Put it in a separate envelope/jar\n- This builds the habit without stress\n\n**The Invisible Savings Method:**\n- Round up every purchase (buy coffee for $3.50? Save $0.50)\n- Save every $5 bill you get\n- Tax refund? Birthday money? Straight to savings\n- Sell one thing you don't use\n\n**Find Hidden Money:**\n- Review all subscriptions (average person has 12!)\n- One less takeout meal = $15-30/month\n- Generic brands = 20-30% savings\n- Negotiate one bill (phone, internet, insurance)\n\n**Automate:**\n- Set up $10/week auto-transfer right after payday\n- You won't miss what you don't see\n- That's $520/year!\n\n**Goal:** \n$500 starter emergency fund in 6 months, then build to $1000.\n\nWhat feels most doable for you to start with?"),
            ("user", "I could probably do $10 a week. But what if an emergency happens before I have enough saved?"),
            ("assistant", "That's a valid concern! Here's how to handle it:\n\n**Your Emergency Fund is Still Worth Building Because:**\n- Even $50 can prevent a small problem from becoming a crisis\n- It breaks the stress cycle of being completely broke\n- It's practice for bigger savings goals\n\n**Bridge Strategies While Building:**\n1. **Credit union emergency loan** - Join now, better rates than payday loans\n2. **Payment plans** - Most medical/dental offices offer these\n3. **Side gig list** - Know 2-3 ways to make quick money (food delivery, plasma donation, online tutoring)\n4. **Negotiate first** - Many bills can be reduced just by asking\n\n**The $10/Week Plan:**\n- Week 4: $40 (covers small car repair)\n- Week 12: $120 (covers most copays/medicines)\n- Week 26: $260 (covers many minor emergencies)\n- Week 52: $520 (real breathing room!)\n\n**Psychological Trick:**\nLabel it \"Car repair fund\" or \"Medical fund\" instead of \"Emergency\" - makes it feel more concrete and less likely you'll raid it.\n\nRemember: Having $100 saved makes you more financially secure than 40% of Americans. Start where you are!"),
        ],

        // Relationship Advice
        vec![
            ("user", "My partner and I keep having the same argument about household chores. How do we break this cycle?"),
            ("assistant", "Recurring arguments about chores are rarely just about the chores - they're often about feeling valued and respected. Here's how to break the pattern:\n\n**Step 1: Change the Conversation**\nInstead of: \"You never do the dishes!\"\nTry: \"I feel overwhelmed when I come home to a full sink. Can we figure out a system?\"\n\n**Step 2: Understand Each Other's Standards**\n- You might see crumbs; they might not\n- You might need a clean space to relax; they might not notice mess\n- Neither is wrong, just different\n\n**Step 3: Create a System, Not Rules**\n- List all weekly tasks\n- Each person claims what they hate least\n- Trade off the remaining tasks\n- Set 'completion times' not 'do it now'\n\n**Step 4: Make It Visible**\n- Use a simple app or whiteboard\n- Check off tasks when done\n- Appreciate completion (seriously, say thank you)\n\n**The Real Fix:**\nHave a weekly 10-minute 'team meeting' - not to criticize, but to adjust the system.\n\nWhat specific chores cause the most conflict?"),
            ("user", "It's mainly that I do things immediately and they let things pile up for days. Then I end up doing it anyway because I can't stand it"),
            ("assistant", "Ah, the classic 'different tolerance levels' issue! This is super common. You're not wrong for wanting things clean, and they're not wrong for having different timing. Here's what actually works:\n\n**The Key Insight:**\nYou're playing \"cleanliness chicken\" - and you always lose because your tolerance breaks first.\n\n**Practical Solutions:**\n\n**1. The 'Deadline Agreement':**\n- Dishes: Done by 9pm daily\n- Laundry: Folded within 24 hours\n- Bathroom: Cleaned by Sunday noon\n- They can do it anytime before deadline\n\n**2. The 'Zone Defense':**\n- You own certain spaces (keep them however you want)\n- They own others (you don't touch/comment)\n- Shared spaces have agreed standards\n\n**3. The 'Batch Day':**\n- Saturday 10am = house reset time\n- Do everything together with music/podcast\n- Reward yourselves after\n\n**4. The Nuclear Option:**\n- If financially possible, hire cleaning help\n- Often cheaper than therapy or divorce!\n\n**Most Important:**\nStop doing their tasks. Put their dishes in a bin. Let their laundry sit. It's uncomfortable but necessary for change.\n\nWhich approach feels most workable for your relationship?"),
        ],

        // Academic Help
        vec![
            ("user", "I have a 10-page research paper due in a week and I haven't started. Help!"),
            ("assistant", "Don't panic! A week is totally doable with the right strategy. Let's break this down:\n\n**Day 1 (Today) - 2 hours:**\n- Choose/narrow your topic (30 min)\n- Create thesis statement (30 min)\n- Find 10 sources (1 hour) - use Google Scholar, JSTOR\n\n**Day 2 - 2 hours:**\n- Read/annotate 5 sources\n- Create outline with evidence slots\n\n**Day 3 - 2 hours:**\n- Read/annotate remaining sources\n- Fill evidence into outline\n\n**Day 4-5 - 3 hours each:**\n- Write 3-4 pages per day (rough draft)\n- Don't edit while writing!\n- Just get ideas down\n\n**Day 6 - 2 hours:**\n- Revise for argument flow\n- Add transitions\n- Strengthen intro/conclusion\n\n**Day 7 - 2 hours:**\n- Final proofread\n- Format citations\n- Submit early in the day!\n\n**Pro Tips:**\n- Write the intro last\n- Each body paragraph = 1 argument + 2-3 pieces of evidence\n- Use Pomodoro (25 min work, 5 min break)\n\nWhat's your topic/subject area?"),
            ("user", "It's for my environmental science class on climate change impacts. I'm supposed to pick a specific impact to focus on"),
            ("assistant", "Perfect! Environmental science papers are great because there's so much current research. Here are some focused topics that are specific enough for 10 pages:\n\n**Strong Topic Options:**\n\n1. **Ocean Acidification's Impact on Coral Reef Ecosystems**\n   - Specific, measurable, lots of data\n   - Can include Great Barrier Reef case study\n\n2. **Climate Migration from Pacific Island Nations**\n   - Human interest angle\n   - Combines science with policy\n   - Tuvalu/Kiribati as specific examples\n\n3. **Permafrost Thaw and Methane Release Feedback Loops**\n   - Cutting-edge research\n   - Clear cause-effect chain\n   - Arctic-specific data\n\n4. **Agricultural Shifts in California's Central Valley**\n   - Local/specific region\n   - Economic data available\n   - Water crisis angle\n\n**Your Thesis Formula:**\n\"Climate change is causing [specific impact] in [specific location], which will lead to [specific consequences] unless [specific actions] are taken.\"\n\n**Quick Source Goldmines:**\n- IPCC reports (incredibly citable)\n- NASA Climate Change\n- NOAA Climate.gov\n- Nature Climate Change journal\n\nWhich topic interests you most? I can help you craft a thesis statement right now!"),
        ],

        // Travel Planning
        vec![
            ("user", "I want to travel solo for the first time. I have 2 weeks and $3000. Where should I go?"),
            ("assistant", "Solo travel is life-changing! With 2 weeks and $3000, you have great options. Let me help you choose based on what you're looking for:\n\n**For First-Time Solo Travelers, I Recommend:**\n\n**Option 1: Portugal ($150-200/day)**\n- Super safe and English-friendly\n- Lisbon → Porto → Algarve route\n- Hostels are social, easy to meet people\n- Amazing food, cheap wine\n- Great public transport\n\n**Option 2: Thailand ($70-120/day)**\n- Solo traveler paradise\n- Bangkok → Chiang Mai → Islands\n- Established backpacker route\n- Your budget goes far\n- Incredibly welcoming culture\n\n**Option 3: Scotland ($180-250/day)**\n- No language barrier\n- Edinburgh → Highlands → Isle of Skye\n- Extremely safe\n- Stunning nature + history\n- Great hostel culture\n\n**Questions to Help Decide:**\n- Do you want adventure or relaxation?\n- Cities or nature?\n- Hot or cool weather?\n- How far are you willing to fly?\n\nWhat type of experience are you dreaming of?"),
            ("user", "I think Portugal sounds perfect! But I'm nervous about staying in hostels alone"),
            ("assistant", "Portugal is an excellent choice! And hostel nervousness is totally normal. Here's how to make it comfortable and safe:\n\n**Hostel Tips for Solo Travelers:**\n\n**Booking Strategy:**\n- Choose hostels with 8.5+ ratings on Hostelworld\n- Book female-only dorms if you prefer\n- Look for ones with these keywords: \"solo traveler friendly,\" \"social atmosphere,\" \"free walking tours\"\n- Splurge on a private room every 3-4 nights for recharge time\n\n**Best Portugal Hostels for Solo Travelers:**\n- **Lisbon:** Home Lisbon Hostel (famous for family dinners)\n- **Porto:** Gallery Hostel (art focus, chill vibe)\n- **Lagos:** Rising Cock Hostel (party/social if that's your thing)\n\n**Safety/Comfort Hacks:**\n- Bring a padlock for lockers\n- Pack earplugs and eye mask\n- Choose bottom bunk (easier access)\n- Keep valuables in a small bag you can take to the shower\n- Join hostel activities - instant friends!\n\n**Your Portugal Itinerary:**\n- Days 1-5: Lisbon (include Sintra day trip)\n- Days 6-9: Porto (include Douro Valley)\n- Days 10-14: Lagos/Algarve (beach time!)\n\n**Budget Breakdown:**\n- Flights: $600-900\n- Accommodation: $30-40/night = $450\n- Food: $40/day = $560\n- Transport: $200\n- Activities/cushion: $800+\n\nWant help booking your first few nights?"),
        ],
    ];

    // Generate conversations using templates with variations
    for _i in 0..count {
        let template = templates.choose(&mut rng).unwrap();
        let mut conversation = Vec::new();

        // Randomly select portion of template to use
        let length = rng.gen_range(2..=template.len());

        for (role, content) in template.iter().take(length) {
            conversation.push(Message {
                role: role.to_string(),
                content: content.to_string(),
            });
        }

        // Sometimes add a follow-up
        if rng.gen_bool(0.3) && conversation.last().unwrap().role == "assistant" {
            conversation.push(Message {
                role: "user".to_string(),
                content: match rng.gen_range(0..5) {
                    0 => "Thanks, that's really helpful!",
                    1 => "Can you elaborate on that last point?",
                    2 => "I see. What would you recommend I do first?",
                    3 => "That makes sense. I'll try that approach.",
                    _ => "Great, thank you for the detailed explanation!",
                }
                .to_string(),
            });

            conversation.push(Message {
                role: "assistant".to_string(),
                content: "You're welcome! Feel free to ask if you have any more questions or need clarification on anything.".to_string(),
            });
        }

        conversations.push(ConversationData {
            messages: conversation,
            metadata: std::collections::HashMap::new(),
        });
    }

    // Shuffle to mix different types
    conversations.shuffle(&mut rng);
    conversations
}

pub fn save_example_conversations(
    conversations: &[ConversationData],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(conversations)?;
    std::fs::write(path, json)?;
    Ok(())
}
