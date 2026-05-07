<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { SETTINGS_KEYS } from '$lib/constants';

	interface Props {
		class?: string;
		disabled?: boolean;
	}

	let { class: className = '', disabled = false }: Props = $props();

	let currentConfig = $derived(config());
	let enableThinking = $derived(currentConfig.enableThinking !== false);

	function toggleThinking() {
		settingsStore.updateConfig(SETTINGS_KEYS.ENABLE_THINKING, !enableThinking);
	}
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button
				type="button"
				variant={enableThinking ? 'default' : 'outline'}
				class="h-8 rounded-full px-3 text-xs font-medium"
				{disabled}
				onclick={toggleThinking}
				aria-pressed={enableThinking}
			>
				Think
			</Button>
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>
				{enableThinking
					? 'Thinking ON — click to disable for this and future messages'
					: 'Thinking OFF — click to re-enable'}
			</p>
		</Tooltip.Content>
	</Tooltip.Root>
</div>
